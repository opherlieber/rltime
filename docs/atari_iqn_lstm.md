# Fast Sample Efficient Q-Learning With Recurrent IQN
## Background
I've recently been working on applying RL to modern games with visual inputs. These games typically require at least 2 dedicated CPU cores, and do not have emulation environments like the ALE which speed up the game, so each ENV runs very slowly in 'real time' at around 15 steps per second (60fps with a frame-skip of 4). Even if I could somehow get 256 distributed envs the acting throughput would still be too low to work well with distributed q-learning algorithms such as APEX and R2D2 without costing a huge amount of resources and time.

The current state of the art in discrete action spaces is [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX) which does very high-throughput distributed acting, using an LSTM model (with an LSTM layer between the CNN and FC layers of the classic DQN model). R2D2 essentially solves Atari games, reaching super-human on almost all of them. However it is not sample efficient, requiring billions of acting and training steps to reach these results, and falling short of other algorithms at the typical sample-efficient threshold of 200M (total) environment frames.

In the sample-efficient Atari domain we have:
- [Rainbow](https://arxiv.org/abs/1710.02298) which combines 6 separate DQN improvements each contributing to the final performance.
- [IQN](https://arxiv.org/abs/1806.06923) (Implicit Quantile Networks) is the state of the art 'pure' q-learning algorithm, i.e. without any of the incremental DQN improvements, with final performance still coming close to that of Rainbow.
- [Impala](https://arxiv.org/abs/1802.01561) (Deep Experts Variant) is a multi-actor distributed actor-critic algorithm with off-policy correction which achieves similar sample-efficient results at a very fast training rate, using a deeper and more complex model than the common Q-learning algorithms.

## Recurrent IQN
Here, I propose a 'toned down' sample-efficient version of R2D2, using the same LSTM model, but combining it with IQN and additional rainbow and R2D2 features.

Training is done on batch sizes of 32x20 (32 independent length-20 sequences), compared to the 64x80 used in R2D2. Acting is done on 32 vectorized ENVs, which along with the large batch size (20x larger than Rainbow/IQN which do 32x1 batches) can reach 200M (total) environment frames on Atari in less than a day on a single machine. This allows for quick experimentation and hyperparmameter tuning.

The number of ENVs can possibly be reduced as a trade-off between training time and availability of multiple ENVs.

Full code is available in the [RLtime Library](https://github.com/opherlieber/rltime).

## Resource Complexity Comparison
Following table compares the acting and training complexity of some existing algorithms with the method presented here:

||R2D2|DQN/Rainbow/IQN|Recurrent IQN|
|----|----|----|----|
|Total ENV Frames|~10B|200M|200M|
|Total Steps Trained|~10B|400M|200M|
|Total Batches Trained|2M|12.5M|312K|
|Training Batch Size|5120|32|640|
|Training Time (Atari)|5D|~7-10D|~18H<sup>1</sup>|

<sup>1 Using a single machine with a dedicated P100 or 1080ti GPU and 8+ CPUs</sup>


## Recurrent State With IQN
Using a recurrent model with IQN has some open questions, in particular how to handle the recurrent state across quantile samples. Some possible options considered:
1. Use a separate recurrent state for each sample. This requires a huge amount of memory in the replay buffer, since we are storing state for each transition in replay. Moreover it's not clear what is the correlation between recurrent state and the random quantiles drawn differently at each timestep (Alternatively maybe use the same quantile across the sequence or at least sort them between samples)
2. Repeat/Tile the recurrent state when entering the LSTM, and merge back (Using 'mean'). This solves the memory explosion, however there is no guarantee the recurrent state representation is 'mean friendly'. 2 options are supported for training sequences:
    * Repeat the recurrent state only at the start of the sequence, and merge back at the end.
    * Merge->Repeat the recurrent state at every timestep. This allows the merge/'mean' to participate in the backpropagation which may help the training learn a more 'mean-friendly' state representation. This results in slower training than the previous option.
3. An easier option is to insert the IQN layer after the LSTM layer and not deal with multi-sample recurrent state. This is simpler, and also results in much faster training time than running a multi-sample batch through the LSTM, but might lose some of the advanatage of IQN, in particular since the IQN layer size becomes ~6x smaller (Being applied to the 512 features from the LSTM output, compared to the ~3K features coming from the CNN layer as in the original paper)

RLtime supports both the 2nd and 3rd options above. The atari results presented here are using the simpler 3rd option (With 32 quantile samples). Some experimentation with the 2nd option did not show any improvement (Though this was done with only 8 quantile samples due to increased training complexity, which indirectly affects various hyperparameter choices so it's hard to compare)
## Results
To test the proposed method I use the 'Arcade Learning Environment' for learning to play classic Atari games, which allows for fast experimentation and comparison to existing papers. I use the atari envs provided by [OpenAI GYM](https://github.com/openai/gym).

### Test Setup and Hyperparameters
The following training/evaluation scheme is used for these results, to be as comparable as possible to the existing (sample-efficient) results we are comparing to:
- Environment episode frame limit of 108000 (30 minutes play time, reduced from the default 400K in GYM)
- Default (reduced) action-set, with up to 30 random no-op actions at start of episodes, and no sticky actions
- Episode termination on life-loss
- Frame-skip of 4 (with 2-frame 'max' merging).
- Hyperparameters are fixed for all games
- Evaluation is done on the final policy at 50M acting steps (200M total environment frames), using a fixed random-action epsilon of 0.001, taking the mean undiscounted reward across 100 episodes.

In addition:
- No frame-stacking is done, as this does not seem to give much when using an LSTM, and improves the replay buffer memory footprint which is not currently optimized for frame-stacking (This also acts as a sort of confirmation that the LSTM is indeed learning something in its state representation)
- Per-actor exploration as in the APEX/R2D2 papers, except the base value of 0.4 is decayed to 0.01 across half the training period, with an absolute minimum of 0.01 for all actors. This ensures more actors follow policy more often as training progresses, while still ensuring a minimal 0.01 rate for occasional exploration.
    - This limits the ability to train with fewer ENVs, in particular with 1 ENV, though in this case we can use the classic fixed epsilon decay from 1.0->0.01 which gives similar results on most games.
- Additional linear features are fed to the LSTM (Though it's not clear how much if at all these help):
    - One-hot encoded last action
    - Last reward (clipped)
    - Timestep (Scaled to [0,1] range)
- Enabled Q-Learning features:
    - Multi-Step (N=2) targets
    - Dueling Network Layers
    - Double Q-Learning
- Training frequency is set to '4', i.e. each acted transition is replayed 4 times (on average), compared to the '8' typically used in DQN variants, as 8 increases training time without any noticeable improvement to the results.

Additional features which gave mixed results and are not currently enabled:
- Prioritized replay buffer with same sequence priority weighting as the R2D2 paper, did not seem to make a big difference, though there are many hyper-parameters there which can be further explored.
    - Priority updates were tested using either the final IQN loss values, or the mean absolute sampled td-errors, with the latter performing better overall

- 'Value Function Rescaling' gives worse results in some games and a worse overall picture (Except for 'MsPacman', which reaches SOTA with 'value function rescaling' enabled)
- Recurrent state burn-in did not give conclusive differences

Full hyperparameters can be found in [this config file](https://github.com/opherlieber/rltime/blob/master/rltime/configs/atari_iqn_lstm.json) which can also be used to run these tests.

### Atari Results
#### Training Charts (Average reward of last 100 episodes)
![Charts](charts.png)
#### Final Evaluation
Evaluation is done on the final policy with a fixed 0.001 exploration rate.

|Game|Rainbow|QR DQN|Imapala Deep|IQN|Recurrent IQN|
|----|----|----|----|----|----|
|Alien|9492|4871|15962|7022|**16920**|
|Assault|14199|22012|19148|29091|**40874**|
|Asterix|428200|261025|300732|342016|**572150**|
|Beam Rider|16850|34821|32463|42776|**60867**|
|Breakout|418|742|787|734|**810**|
|Gravitar|1419|995|360|911|**3261**|
|Ms. Pac-Man|5380|5821|**7342**|6349|7184|
|Q*Bert|33818|**572510**|351200|25750|30463|
|Seaquest|15899|8268|1753|**30140**|23938|
|Space Invaders|18789|20972|43596|28888|**58154**|

<sup>Rainbow/QR-DQN/Impala/IQN results are taken from the respective papers.</sup>

Raw data for these results, including final checkpoint, tensorboard data and training/evaluation logs can be found [here](https://console.cloud.google.com/storage/browser/rltime_results/9a2ef8e/atari/). These can be evaluated/rendered/recorded locally using the evaluation options described in the [RLtime readme](https://github.com/opherlieber/rltime/blob/master/readme.md).

These results are preliminary and may change when averaging in additional runs/seeds. More games will be added when available.


