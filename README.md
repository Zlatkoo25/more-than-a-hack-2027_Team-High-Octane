# Ball possession from tracking alone

Team High Octane, More than a Hack 2027, Phase 1: label every frame of one football match as home
possession (`H`), away possession (`A`), dead ball (`D`) or no answer (`X`).

This repository holds everything needed to go from the organisers' data bundle to our submission
file: the code, the pinned dependencies and the three trained models. The method, the training
data and the results on held-out public games are described in [TECHNICAL.md](TECHNICAL.md).

In short: two models score every frame from the tracking alone (gradient-boosted trees on 121
features, and a small transformer over the 22 players and the ball followed by a GRU), their
scores are averaged, and a Viterbi decoder with a price on every label change turns the scores
into whole possessions. Both models are trained only on public games from other matches (3
Metrica sample games and 7 DFL Bundesliga games), copied many times with simulated tracking
damage.

## Reproduce the submitted file

You need Python 3.12 and the organisers' data bundle. A GPU is not needed for this part.

```
git clone https://github.com/Zlatkoo25/more-than-a-hack-2027_Team-High-Octane
cd more-than-a-hack-2027_Team-High-Octane
python -m venv .venv
.venv\Scripts\activate                  # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/predict.py --data "<dataset folder>" --out submission.csv
```

`<dataset folder>` is the folder of the data bundle that contains `tactical/`. The script reads
`tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`, nothing else. It
takes a few minutes and prints one line.

The file it writes is the file we submitted. Its SHA-256 is

```
44318b16b45f05323467284b095dd292da08903521e3b6d2ffbdc609b0b4fe62
```

Check it with `certutil -hashfile submission.csv SHA256` (Windows) or `sha256sum submission.csv`.
We checked this from a fresh clone on Windows 11, once with an RTX 3060 and once with the GPU
switched off; both runs gave this hash.

If the organisers' repository is cloned next to this one (or `MTAH_ORGANISERS_DIR` points at it),
the script also runs their validator on the file and reports the result in its one line.

## Retrain the models from scratch

The models in `models/` were produced by the commands below. On our machine, retraining from a
fresh clone reproduced the boosted model and the first set-model seed bit for bit. GPU training can
still differ slightly on other GPUs, drivers or library builds, which is why the trained models are
part of the repository.

```
git clone https://github.com/metrica-sports/sample-data public_data/metrica
python -m mtah.dfl                        # downloads and converts the 7 DFL games (about 2.6 GB of traffic)
python scripts/train.py v3                # boosted trees, CPU, about 10 minutes
python scripts/train.py set --seed 0      # set model, GPU, about 35 minutes
python scripts/train.py set --seed 1
python scripts/predict.py --data "<dataset folder>" --out submission.csv
```

`python scripts/predict.py --sample 1 --out check.csv` runs the whole pipeline on Metrica sample
game 1 instead of the match, as a check that an installation works without the data bundle.

The training scripts never read the challenge match.

## Layout

```
mtah/
  metrica.py     load the Metrica sample games (tracking through kloppy, events from their files)
  labels.py      Metrica events -> per-frame H/A/D truth, with the "controlled possession" rule
  dfl.py         download and convert the DFL games; truth from the provider's possession flags
  simulate.py    tracking damage: ball gaps, noise, missing players, anonymous tracklets
  training.py    the training set: every public game as several randomly damaged copies
  features.py    tracking-only features: distances, speeds, gaps, touches, team shape
  boosted.py     model 1 (v3): gradient-boosted trees on 121 features
  setnet.py      model 2: transformer over players and ball, then a GRU over time
  decode.py      Viterbi with a price on each label change
  challenge.py   read the challenge tracking
  score.py       hook to the organisers' validator
  paths.py       folder locations, each overridable with an environment variable
scripts/
  train.py         train the three models
  predict.py       label the match and write the submission file
models/          the three trained models (13 MB)
tests/           unit tests for the label rules (pip install pytest; pytest)
```

## Data and licences

No match data is stored here, and none is needed to train.

- Metrica Sports sample data: https://github.com/metrica-sports/sample-data
- DFL / IDSSE open data (CC BY 4.0): Bassek, M., Rein, R., Weber, H. et al. An integrated dataset
  of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195 (2025).
  https://doi.org/10.1038/s41597-025-04505-y
