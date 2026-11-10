# Hyperparameter Experiments

**Member Name:** Mukunzi Ndahiro James

**Focus:** learning rate, exploration (epsilon), plus batch size and gamma

## Setup

- Game: Freeway (`ALE/Freeway-v5`), 100,000 training steps per run.
- Starting point (experiment 1): learning rate 0.0001, gamma 0.99, batch size 32, epsilon 1.0 down to 0.05 over 50,000 steps. Later runs change a few of these at a time.
- How I measure a run:
  - **Score** = mean crossings in the final 10 games of training.
  - **First 15+** = the training step where the chicken first crossed 15 or more times in one game. This shows how fast the run learned.
- For reference: random moves get about 0 crossings, and holding UP the whole game gets about 21.
- A Freeway game is on a timer (about 2,045 steps), so episode length is the same in every run and I don't use it to compare.

Models are in `models/James/`, logs in `logs/James/` and plots in `plots/James/`.

| Exp | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 0.0001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Baseline. Score 21.3, best game 25. Stayed at 0-1 crossings for the first 40k steps while epsilon was high, then jumped once epsilon fell below about 0.2. First 15+ at 51k steps. |
| 2 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Score 21.7, best game 24. A 10x higher learning rate stayed stable and learned a little faster (first 15+ at 47k). A high learning rate did not break learning in Freeway. |
| 3 | 0.0005 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | Score 21.3, best game 24. Almost the same as experiment 2 (first 15+ at 47k). Any learning rate between 0.0005 and 0.001 worked. |
| 4 | 0.001 | 0.99 | 32 | 1.0 | 0.05 | 25,000 | Score 21.0, best game 24. Halving the exploration time made the agent learn about twice as fast (first 15+ at 26.5k against 51k for the baseline), but the final score did not go up. |
| 5 | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 50,000 | Score 21.6, best game 25. Lowering epsilon end to 0.01 barely changed the score, so the 5% random actions were not what kept the score at about 21. First 15+ at 39k. |
| 6 | 0.001 | 0.99 | 32 | 1.0 | 0.01 | 25,000 | Score 21.6, best game 25. Short exploration with almost no randomness at the end gave fast learning (first 15+ at 22.5k). |
| 7 | 0.001 | 0.99 | 64 | 1.0 | 0.01 | 25,000 | Score 20.8, best game 28. Doubling the batch size learned as fast as experiment 6 (first 15+ at 22.5k) and had one game of 28, which is above what pressing UP alone gives, but the average was less steady. Each run also took longer because of the bigger batch. |
| 8 | 0.0005 | 0.99 | 128 | 1.0 | 0.01 | 25,000 | Score 4.1, best game 8. Second worst. A big batch with a lower learning rate learned very slowly, and exploration had already ended at 25k steps, so the agent was stuck with almost no random actions before it had learned that going up gives points. Reward was only starting to rise (2 to 8) when training ended. Training was also about 4x slower per step. |
| 9 | 0.001 | 0.95 | 32 | 1.0 | 0.01 | 25,000 | **Best run.** Score 22.8, best game 28. Lowering gamma to 0.95 gave the highest score and the fastest learning (first 15+ at 20.4k). With a shorter look-ahead the targets were smaller and easier to learn, and the agent seemed to react more to nearby cars, sometimes beating the ~21 from pressing UP only. |
| 10 | 0.00001 | 0.99 | 32 | 1.0 | 0.05 | 50,000 | **Worst run.** Score 0.3. Reward briefly rose to 19 around 55k steps, but only because the untrained network already happened to prefer UP, so fewer random moves meant more UP. The tiny learning rate never learned that UP gives points, and its small updates flipped the preferred action to NOOP. After that the chicken mostly stood still and scored 0. |

## Results overview

| | Experiment | Score | Why |
|---|---|---|---|
| Best | 9 | 22.8 | Fastest learner and highest score |
| Worst | 10 | 0.3 | Learning rate too small to learn anything |
| Second worst | 8 | 4.1 | Big batch + lower learning rate learned too slowly |

## Lessons from my 10 runs

- **Exploration time mattered most for learning speed.** In Freeway the reward only comes at the top of the road, so while epsilon is high the chicken moves randomly and almost never scores. Shorter exploration (25,000 steps instead of 50,000) let the agent start using what it learned about twice as early.
- **The learning rate had to be big enough.** Anything from 0.0001 to 0.001 was stable, and the higher values learned slightly faster. 0.00001 was far too small and failed completely.
- **A big batch with a lower learning rate hurt.** Batch 128 learned too slowly for the short exploration time and was much slower to train.
- **Gamma 0.95 was slightly better than 0.99**, giving the best score and the fastest learning.
- **Every good run levelled off at about 21.** When I tested a trained model, it pressed UP on every step, and pressing UP alone scores about 21. To go beyond that, the agent has to learn to wait for gaps between cars, which needs much longer training.
- I trained every setting a single time with the same seed. Gaps of less than 1 crossing (like 21.0 vs 21.7) could just be chance, but experiments 8 and 10 failed by a wide margin.

## Settings I would use for the final model

Experiment 9: learning rate 0.001, gamma 0.95, batch size 32, epsilon 1.0 to 0.01 over 25,000 steps. It scored the most, reached 15+ crossings the earliest, and trains quicker than the bigger-batch runs.
