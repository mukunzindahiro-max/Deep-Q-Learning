# Hyperparameter Experiments

**Member Name:** Divine Itu

**Focus:** changing one hyperparameter at a time, covering all six (learning rate, gamma, batch size, epsilon start, epsilon end, epsilon decay)

## How my runs were set up

I started from the group's best configuration so far (James's experiment 9): learning rate 0.001, gamma 0.95, batch size 32, epsilon 1.0 to 0.01 over 25,000 steps. That configuration scored 22.8. In each of my 10 runs I changed **only one** of these values, so any difference in the result comes from that one setting.

All runs: Freeway (`ALE/Freeway-v5`), 100,000 training steps, seed 42, trained on a GPU.

- **Score** = mean crossings over the final 10 training games.
- **15+ at** = training step where a game first reached 15 crossings (how quickly the run learned).
- **Spread** = standard deviation of the final 10 games (lower = more consistent).
- Pressing UP for a whole game gives about 21 crossings, and random play gives about 0.

Files: `models/Divine/`, `logs/Divine/`, `plots/Divine/`.

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | **0.003** | 0.95 | 32 | 1.0 | 0.01 | 25,000 | Score 22.5, best game 26, 15+ at 24.5k. Three times the base learning rate was still stable. It learned a little later than the base but ended at almost the same level, so 0.003 is not too high for this game. |
| 2 | **0.0001** | 0.95 | 32 | 1.0 | 0.01 | 25,000 | Score 20.8, best game 24. Ten times lower learning rate gave the second lowest score. Early games looked fine, but rewards jumped around (17, then 1, then 3 crossings) before settling, and it ended below the "only press UP" level. Smaller updates meant it was still learning when training stopped. |
| 3 | 0.001 | **0.9** | 32 | 1.0 | 0.01 | 25,000 | Score 22.8, best game 26, 15+ at 20.4k. Same score as the base (gamma 0.95). Looking a bit less far ahead did not hurt. |
| 4 | 0.001 | **0.8** | 32 | 1.0 | 0.01 | 25,000 | **Best run.** Score 23.2, best game 28, 15+ at 20.4k. Reached 14-20 crossings by episode 8-11, faster than the others with full exploration. A small gamma makes the agent care mostly about the next few steps, like not getting hit by the next car, and those targets are easier to learn. |
| 5 | 0.001 | 0.95 | **16** | 1.0 | 0.01 | 25,000 | Score 22.4, best game 26, 15+ at 24.5k. A smaller batch learned slightly slower, because each update used fewer, noisier samples, but the final level was close to the base. |
| 6 | 0.001 | 0.95 | **64** | 1.0 | 0.01 | 25,000 | Score 22.8, best game 28, 15+ at 20.4k. A bigger batch matched the base score and had one of the best single games. More samples per update made learning smooth, at the cost of more computation. |
| 7 | 0.001 | 0.95 | 32 | **0.5** | 0.01 | 25,000 | Score 22.1, best game 25, spread 1.4 (most consistent). Starting with only 50% random moves gave 10+ crossings from the very first game. It learned fastest early (15+ at 8k) but did not finish higher than the base. |
| 8 | 0.001 | 0.95 | 32 | **0.1** | 0.01 | 25,000 | Score 22.2, best game 26. Scored 20-21 from the very first game, before any learning happened. With almost no exploration, the untrained network (which happens to prefer UP) basically held UP all game. Training did not push it much above that. It never really explored, so it never found anything better than running straight up. |
| 9 | 0.001 | 0.95 | 32 | 1.0 | **0.1** | 25,000 | **Worst run.** Score 19.9, best game 24, lowest final-10 game 16. Keeping 10% random moves until the end cost about 3 crossings against the base. In Freeway one random DOWN or wait at the wrong moment gets the chicken hit and pushed back. |
| 10 | 0.001 | 0.95 | 32 | 1.0 | 0.01 | **10,000** | Score 21.2, best game 24. Very short exploration made learning very fast (18-20 crossings by episode 4-5, 15+ at 8k), but the final score was lower than with 25,000 steps of exploration. It settled on running up early and had less varied experience to improve from. |

## Best and worst

- **Best:** experiment 4 (gamma 0.8), score 23.2, also the highest score in the group so far.
- **Worst:** experiment 9 (epsilon end 0.1), score 19.9.

## What each hyperparameter did

- **Learning rate:** 0.001 to 0.003 both worked well. 0.0001 was too slow for 100,000 steps and finished lower.
- **Gamma:** lower was slightly better. 0.8 gave the top score, and 0.9 equalled 0.95. Short-term targets seem easier to learn here, because what matters most is avoiding the car right in front of the chicken.
- **Batch size:** small effect. 16 was a little slower, and 64 matched 32.
- **Epsilon start:** starting low (0.5 or 0.1) gave high scores almost immediately, but only because the starting network already liked UP. The final scores were not better than starting at 1.0.
- **Epsilon end:** this mattered. Ending at 0.1 instead of 0.01 gave the lowest score of all my runs, because random moves late in training get the chicken hit.
- **Epsilon decay:** a very short decay (10,000) learned fastest but ended lower. 25,000 was a better balance between exploring and exploiting.

Every run used one seed, so differences under about 1 crossing (for example 22.4 against 22.8) may just be noise. The drop in experiment 9 (about 3 crossings) is large enough to be a real effect.

## My pick for the final model

Learning rate 0.001, **gamma 0.8**, batch size 32, epsilon 1.0 to 0.01 over 25,000 steps (experiment 4). It had the highest score and learned among the fastest, and it only changes one value from the group's previous best.
