# Models

These files are **model data**, not source code. They are licensed separately
from the rest of this repository.

## Contents

| file | contents |
|---|---|
| `model7_weights.npz` | madmom `downbeats_blstm_7.pkl` converted to the npz layout used by `network.py` |
| `model8_weights.npz` | madmom `downbeats_blstm_8.pkl` converted to the same layout |
| `filterbanks.npz` | the three log filterbanks (1024/2048/4096 FFT sizes) used by `features.py` |

## Provenance

- `model7` and `model8` are madmom's public downbeat BLSTM folds 7 and 8 from
  [CPJKU/madmom_models](https://github.com/CPJKU/madmom_models), files
  `downbeats/2016/downbeats_blstm_7.pkl` and `_8.pkl`.
- The committed weights were extracted from rekordbox's bundled
  `detect_beat/model7` / `model8` SavedModels and verified to be float-for-float
  identical to the public madmom folds. `fetch_models.py` reproduces the same
  arrays directly from the public pkl files.
- `filterbanks.npz` contains the matrices for 3/6/12 bands per octave,
  fmin 30 Hz, fmax 17000 Hz, normalized, unique filters. They match madmom's
  `LogarithmicFilterbank` and the matrices embedded in rekordbox's SavedModel.

## License

All files in this directory are distributed under
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**
(CC BY-NC-SA 4.0). See `LICENSE-CC-BY-NC-SA-4.0.txt`.

You are free to share and adapt the files for non-commercial purposes, provided
you give appropriate credit, link to the license, and distribute adaptations
under the same license. Commercial use requires permission from the original
rights holders (CPJKU / Johannes Kepler University Linz).

Attribution: Department of Computational Perception, Johannes Kepler University
Linz, and Austrian Research Institute for Artificial Intelligence (OFAI).
Upstream: https://github.com/CPJKU/madmom_models
