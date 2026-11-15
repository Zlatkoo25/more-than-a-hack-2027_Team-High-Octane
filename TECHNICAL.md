# Ball possession from tracking alone: technical document

More than a Hack 2027, Phase 1. Team: High Octane. Repository: https://github.com/Zlatkoo25/more-than-a-hack-2027_Team-High-Octane

## 1. Approach

Every frame gets H, A or D from two learned models whose per-frame scores are averaged and then
decoded as whole possessions.

1. **v3**: gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) on 121
   features computed from the tracking: ball position, speed and gaps; distances from the ball to
   each team's nearest players; touches and carries; player speeds (a whole-pitch slowdown marks a
   stoppage); and team shape (centroid, block length and width, defensive line, forward
   velocity), which names the team in possession even when the ball is not seen.
2. **Set model**: a small transformer reads each frame's 22 players and the ball as an unordered
   set (players attend to the ball and to each other), then a two-layer bidirectional GRU runs
   over the match at 5 Hz. It is averaged over two training seeds and four views of the match
   (as is, pitch mirrored, teams swapped with H and A swapped back, both).
3. **Decoding**: Viterbi over H, A and D with a price of 25 (in log-probability units) on every
   label change. The price sets how many possessions come out, which is what `spellF1` rewards;
   averaging frames or majority smoothing cannot do that.
4. **Scored spans**: frames outside 11,185-90,651 and 110,808-190,197 are written `X`; frames
   110,808-113,741 are held to `D`, as the organisers' primer states.

**Why this design.** Three choices carry most of the result, each measured on held-out public
games (section 6):

- *Training on damaged tracking.* The match tracking is automatic and uncleaned. Every training
  game is copied many times with simulated damage (ball dropouts in bursts biased to fast balls,
  position noise, missing players, anonymous track fragments, wrong-team slots, a hidden dead
  ball), so the models learn to work from partial evidence instead of assuming a clean ball.
- *Two providers' conventions in training.* Metrica's data has no ball position while play is
  dead; DFL's tracks the ball throughout. Models trained on one alone misread the other (0.556
  instead of about 0.78 on DFL); trained on both, they do not care.
- *Possessions as objects.* The decoder works on whole possessions with a price per change, and
  every evaluation counts possessions with the organisers' own scorer, not sampled frames.

## 2. Assets used and dropped

| asset | used? | how, or why not |
|---|---|---|
| Tactical tracking (ATD) | yes | the only input at prediction time |
| Smart Tagging | tried, dropped | votes on the team inside tagged phases; on simulated tags at five quality levels the effect was within ±0.007 spellF1 and did not grow with tag quality |
| Homography, tactical video, broadcast feed | no | not needed for a tracking-only method |
| Metrica Sports sample games 1-3 | yes, training | public, with event data |
| DFL / IDSSE Bundesliga open data, 7 games | yes, training | CC BY 4.0, Bassek et al., *Scientific Data* 2025 |
| SkillCorner open data, 10 A-League games | tried, dropped | MIT licence. Broadcast tracking with automatic phase labels; it measured real camera damage (used to shape the simulator), but as training data it taught its own labelling convention and cost 0.01-0.015 elsewhere |

## 3. Training data

No data from the provided match was used for training, tuning or choosing any setting.

| dataset | labels from | amount |
|---|---|---|
| Metrica sample games 1-3 | our code (`mtah/labels.py`), from Metrica's published event data | 3 matches, 8 damaged copies each |
| DFL / IDSSE, 7 Bundesliga matches | the provider's `BallStatus` and `BallPossession` flags, set by human operators | 7 matches, 4 damaged copies each |

Each copy is also used with the teams swapped (labels swapped to match): 104 training sequences
in all. Sources: github.com/metrica-sports/sample-data; the IDSSE open data release (Bassek et
al. 2025).

## 4. The possession rules

The models learn possession from the labels above, so the rules are the ones those labels encode:

- **Control, not contact.** A team gains possession only with control: it keeps the ball for
  2 s or completes a pass. Touches, deflections and challenges without control do not change
  possession. This follows CHALLENGE.md's "keeps the ball for a few seconds" and was also the
  best match to the DFL operators' labels (0.784 at 2 s against 0.701 at 1 s and 0.764 at 3 s).
- **Loose balls and passes in flight** belong to the team that last had control until the other
  team controls the ball.
- **Dead ball** starts when the ball crosses a line, or, for a foul, offside or handball, at the
  last touch before the restart (play often runs on for a moment before the whistle). It ends
  at the restart, and the restarting team has the ball from then.
- **Restarts** are not a hand-written rule; the models learn them from the labelled games.

## 5. Settings and where each came from

| setting | value | source |
|---|---|---|
| change price | 25 | held-out Metrica and DFL games; flat between 15 and 40 |
| set-model weight in the average | 0.5 | held-out games; flat between 0.35 and 0.5 |
| set model | 2 transformer layers (dim 64), GRU hidden 192, 5 Hz, 20 epochs | held-out games |
| seeds × views | 2 × 4 | held-out games (+0.007 Metrica, +0.013 DFL over one of each) |
| control rule | 2 s | CHALLENGE.md, confirmed on DFL operator labels |
| scored spans, forced D before second kick-off | as published | organisers' PRIMER.md |

## 6. Results on held-out public games

spellF1 from the organisers' scorer on games the models did not train on: Metrica
leave-one-game-out (6 damage scenarios each) and 3 held-out DFL games (5 scenarios each).

| method | Metrica | DFL |
|---|---|---|
| organisers' baseline (re-implemented) | 0.19-0.44 by scenario | |
| v3 alone, Metrica + DFL training | 0.758 | 0.684 |
| set model alone | 0.781 | 0.753 |
| **v3 + set model, 2 seeds × 4 views (submitted)** | **0.806** | **0.782** |

macroF1 for the submitted recipe: 0.947 and 0.935. Trained on DFL alone and tested on Metrica, a
provider it never saw, the recipe scores 0.782 instead of 0.799, so an unseen provider costs
about 0.02 once both dead-ball conventions are covered.

**Cost.** The three trained models total 13 MB. Training takes about 35 minutes per set-model
seed on one consumer GPU (RTX 3060, throttled to 75% use, under 4 GB of memory) and minutes for
v3 on CPU; labelling the match takes minutes. No video is decoded.

## 7. What did not help (all on held-out public games)

| idea | effect on spellF1 |
|---|---|
| Smart Tagging votes (simulated tags, five quality levels) | -0.003 to +0.007, unrelated to tag quality |
| a turnover detector lowering the change price | perfect detector +0.015; realistic about +0.005 |
| separate prices for team and dead-ball changes | within 0.004 |
| loss weighted towards short possessions | +0.002 / 0.000 inside the ensemble |
| set model at 12.5 Hz instead of 5 Hz | -0.004 / -0.011 in the mix |
| team-style prior (true ball share and spell lengths) | at most +0.003, even as an oracle |
| imputing missing players from formations | +0.004 even with perfect imputation |
| self-training on the target game's own confident frames | -0.010 to 0.000 |
| larger boosted model, stacking, alternative decoders | within 0.005 |
| decoding for the best expected spellF1 (60 sampled sequences) instead of the most probable sequence | +0.001 on average, within 0.008 either way |
| the touch sequence from tracking: as features, as an explicit control rule, or to propose extra short possessions for a second look | within 0.01, even with a perfect second look |

## 8. Limitations

- **Short possessions.** About half the true possessions of 2-4 s are missed (given to the other
  team or merged into a neighbour). With the true team supplied, spellF1 would rise from about
  0.80 to 0.97, so the team call in brief possessions is where nearly all the remaining error
  is. Decoder changes, loss weighting, finer time steps, phase tags and touch-sequence features did
  not move it.
- **Stand-in truth.** The Metrica labels are derived by our code from event data, not drawn by an
  operator. DFL's are operator labels but a different league and provider.
- **Domain.** The models have seen two providers, both with cleaner tracking than the match's,
  which is automatic and uncorrected (the organisers' DATA.md reports the ball missing on 41% of
  graded frames). The damage simulation is there to cover this, and on held-out public games the
  miss rate was flat (23-26%) across damage levels, but it cannot be proven on the match itself.
- **With more time:** a second output on the set model that points at the player in control,
  trained on the event data that comes with the public games; a model that outputs possessions
  directly as (start, end, team) and is trained on their overlap with the true ones, as spellF1
  scores them; the tactical video at the moments the models are unsure, the one source that
  holds information the tracking lacks; and more operator-labelled matches from more providers.

## 9. AI assistance

The code and documents were written with the help of an AI coding assistant. It helped with:

- writing the code: data loaders for the public games, the damage simulation, the features, the
  boosted and set models, training, the decoder and the prediction script;
- designing and running the experiments on public games behind sections 6 and 7, and reading
  their results;
- comparing the approach with the published literature;
- drafting this document.

The choices of what to try, what to keep and what to submit were the team's.

## 10. Reproduction

The repository holds the code, the pinned dependencies and the three trained models. Python 3.12.

To regenerate the submitted file (no GPU needed, a few minutes):

    python -m venv .venv ; .venv\Scripts\activate
    pip install -r requirements.txt
    python scripts/predict.py --data "<folder holding tactical\>" --out High_Octane.csv

This gives the submitted file exactly: SHA-256
`44318b16b45f05323467284b095dd292da08903521e3b6d2ffbdc609b0b4fe62`. It was checked from a fresh
clone, with and without the GPU.

To retrain the models from the public games (README.md has the details):

    git clone https://github.com/metrica-sports/sample-data public_data/metrica
    python -m mtah.dfl                         # downloads the DFL open data
    python scripts/train.py v3
    python scripts/train.py set --seed 0       # GPU
    python scripts/train.py set --seed 1

On our machine, retraining from a fresh clone reproduced all three models: both set-model files
bit for bit, and a boosted model that gives identical scores. GPU training can still differ
slightly on other GPUs, drivers or library builds, which is why the trained models are in the
repository.

The prediction script reads the match tracking, writes the submission and prints one line. No
match data is stored in the repository. The repository contains the code behind the submitted
file only; the experiment scripts behind the comparisons in sections 6 and 7 are not part of it.
