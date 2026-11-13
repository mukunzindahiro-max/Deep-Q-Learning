# Hyperparameter Experiments

**Member Name:** Larissa

**Game:** Freeway (`ALE/Freeway-v5`)

Each experiment uses 100,000 training steps, seed 42, and the default replay/training settings from `train.py` (buffer 50,000, learning starts at 10,000, train every 4 steps, target update every 1,000). Every configuration is different from all 20 of James's and Divine's runs.

- **Score** = mean crossings over the final 10 training games.
- **15+ at** = training step where a game first reached 15 crossings (how fast it learned).
- **Spread** = standard deviation of the final 10 games (lower = more consistent).
- Pressing UP for a whole game gives about 21 crossings; random play gives about 0.

Files: `models/Larissa/`, `logs/Larissa/`, `plots/Larissa/`.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0002 | 0.99 | 32 | 1 | 0.05 | 75,000 | Not run |
| 2 | 0.0007 | 0.97 | 32 | 1 | 0.02 | 35,000 | Not run |
| 3 | 0.002 | 0.9 | 32 | 0.8 | 0.01 | 25,000 | Not run |
| 4 | 0.001 | 0.7 | 32 | 1 | 0.01 | 25,000 | Not run |
| 5 | 0.001 | 0.5 | 32 | 1 | 0.01 | 25,000 | Not run |
| 6 | 0.001 | 0.8 | 64 | 1 | 0.01 | 15,000 | Not run |
| 7 | 0.001 | 0.8 | 16 | 0.3 | 0.01 | 25,000 | Not run |
| 8 | 0.0005 | 0.8 | 32 | 1 | 0.03 | 40,000 | Not run |
| 9 | 0.001 | 0.8 | 32 | 1 | 0.2 | 25,000 | Not run |
| 10 | 0.0003 | 0.95 | 64 | 0.8 | 0.01 | 20,000 | Not run |

## What each hyperparameter did

_Write this after you have looked at the plots in `plots/Larissa/`. Compare experiments that differ in one or two values, for example 4 vs 5 vs 9 (gamma 0.7 / 0.5 / 0.8 at otherwise similar settings) and 9 vs the others (epsilon end 0.2)._

