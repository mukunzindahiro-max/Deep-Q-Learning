# Hyperparameter Experiments

**Member Name:** Larissa

**Game:** Freeway (`ALE/Freeway-v5`)

Each experiment uses 100,000 training steps, seed 42, and the default replay/training settings from `train.py` (buffer 50,000, learning starts at 10,000, train every 4 steps, target update every 1,000). Every configuration is different from all 20 of James's and Divine's runs. I used their results as a starting point and tested values nobody in the group had tried yet: gamma below 0.8, epsilon end of 0.2, longer and shorter epsilon decay, and learning rates between their values.

- **Score** = mean crossings over the final 10 training games.
- **15+ at** = training step where a game first reached 15 crossings (how fast it learned).
- **Spread** = standard deviation of the final 10 games (lower = more consistent).
- Pressing UP for a whole game gives about 21 crossings; random play gives about 0.

Files: `models/Larissa/`, `logs/Larissa/`, `plots/Larissa/`.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.0002 | 0.99 | 32 | 1 | 0.05 | 75,000 | Score 20.2, best game 22, 15+ at 57k (slowest of my runs). Stayed at 0-1 crossings for the first 40k steps because epsilon was still high and the chicken moved randomly. Long exploration plus a small learning rate left too little time to improve once epsilon got low. |
| 2 | 0.0007 | 0.97 | 32 | 1 | 0.02 | 35,000 | Score 21.7, best game 25, 15+ at 35k. Steady middle-of-the-road run. Shorter exploration than experiment 1 let it start scoring about 20k steps earlier and finish higher. |
| 3 | 0.002 | 0.9 | 32 | 0.8 | 0.01 | 25,000 | Score 21.7, best game 26, spread 1.6, 15+ at 24.5k. A high learning rate (0.002) was stable. Starting epsilon at 0.8 gave a few crossings (3-10) already in the first games, before learning began. |
| 4 | 0.001 | 0.7 | 32 | 1 | 0.01 | 25,000 | Score 20.9, best game 26, spread 3.4 (least consistent of the runs that learned). Learned fast (15+ at 20k) and reached 25-26 in some games, but one final game dropped to 13. Gamma 0.7 made the scores jump around more. |
| 5 | 0.001 | 0.5 | 32 | 1 | 0.01 | 25,000 | **Best run.** Score 22.8, best game 26, 15+ at 16k. Even with gamma 0.5 the agent learned quickly and scored above the "only press UP" level. In Freeway the reward comes every time the chicken reaches the top, so the agent does not need to plan far ahead. The final 10 games have the same scores as James's experiment 9: by the end both agents press UP almost every step, and the shared seed gives the same car patterns, so the same play gives the same scores. |
| 6 | 0.001 | 0.8 | 64 | 1 | 0.01 | 15,000 | Score 22.0, best game 25, 15+ at 14k (fastest of the runs that started at epsilon 1.0). Very short exploration plus a bigger batch made the agent use what it learned early, and the bigger batch kept updates smooth. |
| 7 | 0.001 | 0.8 | 16 | 0.3 | 0.01 | 25,000 | Score 22.1, best game 27. Scored 16-21 from the first games, before any learning, because with only 30% random moves the untrained network (which happens to prefer UP) mostly ran straight up. Training kept it at that level but did not push it much higher. Low exploration gives quick results but little chance to discover anything better. |
| 8 | 0.0005 | 0.8 | 32 | 1 | 0.03 | 40,000 | Score 22.0, best game 27, spread 1.4 (most consistent run). Slower start (15+ at 30.7k) because of the longer exploration and lower learning rate, but very steady at 20-24 crossings in the final games. |
| 9 | 0.001 | 0.8 | 32 | 1 | 0.2 | 25,000 | **Worst run.** Score 17.3, best game 22, spread 3.9, one final game of only 7. It learned at a normal speed (15+ at 26.6k) but keeping 20% random moves until the end kept pulling the score down. In Freeway a random DOWN or wait at the wrong moment gets the chicken hit and pushed back. |
| 10 | 0.0003 | 0.95 | 64 | 0.8 | 0.01 | 20,000 | Score 20.3, best game 23, 15+ at 80k (latest of all). Started at 2-14 crossings (the untrained network preferred UP), then collapsed to 0-4 crossings once learning began at 10k steps: the early updates changed the network's preferred action away from UP. With a low learning rate it took until about 75k steps to recover, and it reached 19-23 only in the last games. |

## Results overview

| | Experiment | Score | Why |
|---|---|---|---|
| Best | 5 | 22.8 | Learned fast (15+ at 16k) and scored above pressing UP only |
| Most consistent | 8 | 22.0 | Spread of only 1.4 in the final 10 games |
| Worst | 9 | 17.3 | 20% random moves until the end kept getting the chicken hit |
| Second worst | 1 | 20.2 | Exploration lasted too long (75k steps) for a 100k-step run |

## What each hyperparameter did

- **Gamma:** Freeway did not need a high gamma. 0.5 (experiment 5) gave my best score, and 0.8-0.97 all scored 21.7-22.1. 0.7 (experiment 4) was less consistent, so lower gamma is not always better, but nothing below 0.95 hurt the final score.
- **Epsilon end:** this mattered most. Ending at 0.2 (experiment 9) gave the lowest score of all my runs, about 5 crossings worse than ending at 0.01. Together with Divine's run at 0.1, this shows the more randomness is left at the end, the lower the score.
- **Epsilon decay:** long decay slowed learning. 75k steps (experiment 1) reached 15+ crossings at 57k steps, while 15k steps (experiment 6) reached it at 14k. Shorter exploration means the agent starts using what it learned earlier.
- **Epsilon start:** starting low (0.3 in experiment 7) gave high scores from the first game, but only because the untrained network already liked UP. It did not finish higher than runs that started at 1.0.
- **Learning rate:** 0.0005-0.002 all worked. 0.0003 with batch 64 (experiment 10) and 0.0002 with long exploration (experiment 1) were the slowest to learn.
- **Batch size:** small effect. 16 and 64 both reached about 22 when the other settings were good.

Every run used one seed, so differences under about 1 crossing (for example 21.7 against 22.0) may just be noise. The drop in experiment 9 (about 5 crossings) and the slow learning in experiments 1 and 10 are large enough to be real effects.

## My pick for the final model

Experiment 5: learning rate 0.001, gamma 0.5, batch size 32, epsilon 1.0 to 0.01 over 25,000 steps. It had my highest score and learned among the fastest. Experiment 8 is a close second if consistency matters more than the top score.
