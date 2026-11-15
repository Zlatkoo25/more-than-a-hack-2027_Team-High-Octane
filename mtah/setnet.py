"""The set model: each frame's players and ball are tokens that attend to each other (a small
transformer), then a bidirectional GRU runs over the match at 5 Hz.

Attention relates every player to the ball and to nearby opponents directly, and empty slots are
masked out, so the model works on anonymous tracklets and on frames with players or the ball
missing.

Two optional limits for a shared machine, both environment variables:
    MTAH_GPU_GB     cap on the GPU memory PyTorch may allocate, in GiB (default 6.5)
    MTAH_GPU_DUTY   share of time the GPU may be kept busy (default 0.75; 1 = no pauses)
Neither changes any result.
"""

import os
import time

import numpy as np
import torch
from torch import nn

from . import features

STEP = 5                       # frames per model step: 5 Hz
CHUNK, BATCH = 600, 16         # training windows: 600 steps (2 minutes) each, 16 per batch
EPOCHS = 20
DIM, HIDDEN = 64, 192
N_INPUTS = 117
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GPU_GB = float(os.environ.get("MTAH_GPU_GB", 6.5))
GPU_DUTY = float(os.environ.get("MTAH_GPU_DUTY", 0.75))
if DEVICE.type == "cuda":
    _total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(GPU_GB * 2**30 / _total, 0.75))


def rest(started):
    """Pause after a block of GPU work that began at time.time() == started."""
    if DEVICE.type == "cuda" and GPU_DUTY < 1.0:
        torch.cuda.synchronize()
        time.sleep((time.time() - started) * (1.0 / GPU_DUTY - 1.0))


def gpu_peak():
    """Peak memory PyTorch reserved on the GPU so far, in GiB (0 on CPU)."""
    return torch.cuda.max_memory_reserved() / 2**30 if DEVICE.type == "cuda" else 0.0


def raw_design(ball, players, team, period, swap):
    """(T, 117): own team then other team (11 players sorted along the pitch: x, y, vx, vy,
    present), then the ball (x, y, vx, vy, visible, distance to each team). Everything is in the
    'own' team's attacking frame, so swapping teams is an exact relabelling."""
    own = 1 if swap else 0
    with np.errstate(all="ignore"):
        cx = {k: np.nanmean(players[:, team == k, 0], axis=1) for k in (0, 1)}
    d = features.attack_direction(cx[own], cx[1 - own], period)[:, None]
    x = (players[:, :, 0] - features.PITCH[0] / 2) * d
    y = (players[:, :, 1] - features.PITCH[1] / 2) * d
    v = features.velocity(players, 2) * d[:, :, None]
    cols = []
    rows = np.arange(len(ball))[:, None]
    for k in (own, 1 - own):
        idx = np.flatnonzero(team == k)
        xs = x[:, idx]
        order = np.argsort(np.where(np.isnan(xs), np.inf, xs), axis=1)[:, :11]
        present = ~np.isnan(xs[rows, order])
        for arr, scale in ((xs, 52.5), (y[:, idx], 34.0), (v[:, idx, 0], 8.0), (v[:, idx, 1], 8.0)):
            cols.append(np.nan_to_num(arr[rows, order]) / scale)
        cols.append(present.astype(np.float32))
    vis = ~np.isnan(ball[:, 0])
    bv = features.velocity(ball, 2) * d
    dist = features.team_ball_distance(ball, players, team)[:, [own, 1 - own]]
    cols.append(np.column_stack([
        np.nan_to_num((ball[:, 0] - features.PITCH[0] / 2) * d[:, 0]) / 52.5,
        np.nan_to_num((ball[:, 1] - features.PITCH[1] / 2) * d[:, 0]) / 34.0,
        np.nan_to_num(bv) / 20.0, vis.astype(np.float32),
        np.nan_to_num(np.clip(dist, 0, 30) / 30.0, nan=1.0)]))
    return np.concatenate(cols, axis=1).astype(np.float32)


class SetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.player = nn.Linear(10, DIM)
        self.ball = nn.Linear(7, DIM)
        layer = nn.TransformerEncoderLayer(DIM, 4, 128, dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.gru = nn.GRU(3 * DIM, HIDDEN, num_layers=2, bidirectional=True, batch_first=True,
                          dropout=0.2)
        self.out = nn.Linear(2 * HIDDEN, 3)
        side = torch.zeros(22, 2)
        side[:11, 0] = 1.0
        side[11:, 1] = 1.0
        self.register_buffer("side", side)

    def frames(self, flat):
        """(n, 117) frames -> (n, 3 * DIM): each frame's players and ball, attended and pooled."""
        n = len(flat)
        pl = flat[:, :110].reshape(n, 2, 5, 11).permute(0, 1, 3, 2).reshape(n, 22, 5)
        ball = flat[:, 110:]
        present = pl[:, :, 4] > 0.5
        vis = ball[:, 4:5]
        rel = (pl[:, :, :2] - ball[:, None, :2]) * vis[:, :, None]
        dist = rel.norm(dim=2, keepdim=True)
        feats = torch.cat([pl[:, :, :4], self.side.expand(n, 22, 2), rel, dist,
                           vis[:, :, None].expand(n, 22, 1)], dim=2)
        tokens = torch.cat([self.player(feats), self.ball(ball)[:, None]], dim=1)
        pad = torch.cat([~present, torch.zeros(n, 1, dtype=torch.bool, device=flat.device)], dim=1)
        h = self.enc(tokens, src_key_padding_mask=pad)
        w = present.float()[:, :, None]
        own = (h[:, :11] * w[:, :11]).sum(1) / w[:, :11].sum(1).clamp(min=1)
        other = (h[:, 11:22] * w[:, 11:]).sum(1) / w[:, 11:].sum(1).clamp(min=1)
        return torch.cat([own, other, h[:, 22]], dim=1)

    def forward(self, x):
        b, t, _ = x.shape
        frame = self.frames(x.reshape(b * t, -1)).reshape(b, t, -1)
        return self.out(self.gru(frame)[0])


def fit(seqs, seed=0):
    """seqs: list of (X float16 (n, 117) at 5 Hz, y int64 (n,)). The training data stays in
    system memory; only each batch is copied to the GPU."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = [(torch.from_numpy(X), torch.from_numpy(y)) for X, y in seqs]
    net = SetNet().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * sum(len(X) // CHUNK for X, _ in data) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps + 8)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        net.train()
        chunks = []
        for i, (X, _) in enumerate(data):
            off = int(rng.integers(0, CHUNK))
            chunks += [(i, a) for a in range(off, len(X) - CHUNK, CHUNK)]
        rng.shuffle(chunks)
        for b in range(0, len(chunks) - BATCH + 1, BATCH):
            xb = torch.stack([data[i][0][a:a + CHUNK] for i, a in chunks[b:b + BATCH]])
            yb = torch.stack([data[i][1][a:a + CHUNK] for i, a in chunks[b:b + BATCH]])
            started = time.time()
            xb = torch.clamp(torch.nan_to_num(xb.to(DEVICE, non_blocking=True).float()), -6, 6)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(net(xb).reshape(-1, 3), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            rest(started)
    return net


def predict(net, X, n, chunk=4000):
    """(n, 3) log-probabilities for H, A, D at 25 Hz from X (T, 117) at 5 Hz. The per-frame
    attention runs chunk by chunk (bounded GPU memory); the GRU then reads the whole match."""
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.clip(np.nan_to_num(X.astype(np.float32)), -6, 6))
        parts = []
        for i in range(0, len(x), chunk):
            started = time.time()
            parts.append(net.frames(x[i:i + chunk].to(DEVICE)))
            rest(started)
        seq = torch.cat(parts)[None].contiguous()
        try:
            hidden = net.gru(seq)[0]
        except RuntimeError:          # cuDNN can refuse a very long sequence
            with torch.backends.cudnn.flags(enabled=False):
                hidden = net.gru(seq)[0]
        logp = torch.log_softmax(net.out(hidden)[0], dim=1).cpu()
    return np.repeat(logp.numpy(), STEP, axis=0)[:n]
