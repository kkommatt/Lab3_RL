# CarRacing-v3: improved DQN

Improvement of the baseline from
[PavloZakala/ReinforcementLearning](https://github.com/PavloZakala/ReinforcementLearning)
(`dqn_car_racing.py`, 1000 training iterations).

## Result

Evaluated with the unchanged `test_car_racing.py`, 50 episodes, greedy policy:

| | Baseline | This version | Change |
|---|---|---|---|
| Mean | 840.6 | **904.4** | +63.8 (+7.6%) |
| Std | 103.2 | **37.9** | -63% |
| Min | 470.4 | **743.9** | +273.5 |
| Max | 912.7 | **933.0** | +20.3 |

Additional statistics of this version (computed from the 50 episode rewards): median 915.7;
36 of 50 episodes scored >= 900; 46 of 50 scored >= 850; only 2 episodes were below 800
(743.9 and 752.2).

The main gain is **reliability**: the average went up by 7.6%, but the spread dropped by almost
two thirds and the worst episode improved by 273 points. The baseline's weak episodes
(down to 470) largely disappeared.

Evaluation environment: Windows, Python 3.13, gymnasium 1.4.0, torch 2.11 (CUDA 12.8).
Training command: `python train.py --total-steps 2000000` (adjust this line if your run differed).

## What is unchanged (task constraints)

- Environment settings: `gym.make("CarRacing-v3", continuous=False)` + grayscale, resize to 84x84,
  stack of 4 frames. The code is the same as in the baseline (`make_env`, now in `car_env.py`).
- `test_car_racing.py` is byte-identical to the original.
- The interface used by the test: `make_env()`, `QNetwork(a_size=...)`, `QNetwork.act(state)`,
  `device`, checkpoint as a `state_dict`.

## What was changed, how, and why

The reasons below come from reading the baseline code and from a diagnostic of the baseline
checkpoint (see "Diagnosis"). Each row is a design rationale, not a measured contribution:
no ablation study was run (see "Limitations").

| # | Change | How (in code) | Why |
|---|---|---|---|
| 1 | **Double DQN** | `DQNAgent.replay`: the online net picks the `argmax` action of the next state, the target net evaluates it | Plain DQN takes the `max` of the target net, which systematically overestimates Q-values; overestimated values make a policy prefer risky actions |
| 2 | **3-step returns** | `FrameReplayBuffer` accumulates `r_t + g*r_{t+1} + g^2*r_{t+2}` and bootstraps with `g^k` | Rewards are small and frequent (-0.1 per frame, about +3 per tile). One-step bootstrapping passes information about a good turn back to earlier actions slowly; 3 steps propagate it faster |
| 3 | **Correct handling of time limits** | `terminated` and `truncated` are separated: only a real termination zeroes the future value, a truncation still bootstraps | The baseline stores `done = terminated or truncated`, so an episode cut at the 1000-frame limit is treated as if the future were worth 0, which biases the targets |
| 4 | **Dueling head** | `QNetwork`: `Q = V(s) + A(s,a) - mean(A)` | In many frames the choice of action barely matters (straight road); the network can learn the state value without relearning it for every action |
| 5 | **No-progress truncation (training only)** | `NoProgressTruncation` ends a training episode after 100 frames without a new tile | Time on the grass produces nearly identical, uninformative transitions. The wrapper is used only for training; the test environment is untouched. It is a truncation, so values stay unbiased (see #3) |
| 6 | **Frame-level replay buffer** | Each 84x84 frame is stored once; 4-frame stacks are rebuilt on sampling | A transition with two full stacks costs 8 frames of memory, here about 1 frame. 300k transitions fit in about 2.1 GB (baseline: 50k in about 2.8 GB), so the training data is much more diverse |
| 7 | **Soft target update + LR decay** | Polyak averaging `tau = 0.005`; learning rate 1e-4 -> 1e-5; gradient clipping 10; Adam `eps = 1.5e-4` | A periodic hard copy makes targets jump; a decaying learning rate reduces late-training noise, which targets the consistency of the policy from episode to episode |
| 8 | **Step-based epsilon schedule** | Epsilon decays linearly with environment steps (1.0 -> 0.02 over the first 40% of training) | The baseline decayed epsilon per episode, but episode length varies strongly, so the amount of exploration depended on how the agent was doing |
| 9 | **Checkpoint selection by greedy evaluation** | Every 50k steps: greedy evaluation on fixed tracks (seeds 10000+); the top 3 snapshots plus the final network are re-evaluated on fresh tracks (seeds 50000+); the best one is saved | The test script measures a greedy policy, while the baseline selected "best" by the epsilon-greedy training average, which is a different and noisier quantity. Choosing by the tested metric, on tracks not used for the first selection, reduces the chance of saving a lucky snapshot |
| 10 | **Parallel environments that work on Windows** | Env code is in `car_env.py` without torch; `train.py` has a light top level | `AsyncVectorEnv` workers are spawned on Windows and re-import the main script; importing torch with CUDA in 16 workers caused `WinError 1455` (paging file too small) |

Kept from the baseline: initial learning rate 1e-4, gamma 0.99, batch size 128, Nature-DQN
convolutional layers, Huber loss, 8 parallel environments, 2 updates per step.

## Diagnosis of the baseline

A diagnostic run of the baseline checkpoint (8 episodes on my machine, a small sample; the
script is not part of this repository) showed that the poor episodes were **not crashes**: none
ended with the -100 off-playfield penalty. The weak episodes were cut by the 1000-frame limit
with the lap unfinished (for example 201 of 344 tiles). A completed lap scores about
`1000 - 0.1 * frames`, so the loss comes from slow or inaccurate driving that wastes frames.
This is why the changes target faster credit assignment (#2), unbiased value targets near the
time limit (#3), more useful and more diverse data (#5, #6) and a reliable checkpoint choice (#9).

## Project structure

| File | Purpose |
|---|---|
| `train.py` | Entry point (`python train.py ...`) |
| `dqn_car_racing.py` | Network, replay buffer, agent, training loop, evaluation, `render()` |
| `car_env.py` | `make_env` (identical to the baseline), `NoProgressTruncation`, `make_train_env` |
| `test_car_racing.py` | Original test script, unchanged |
| `checkpoint_CarRacing-v3_best.pth` | Trained weights |

## Installation (Windows / Linux)

Use Python 3.12 or 3.13. For Python 3.14 gymnasium requires `box2d-py`, which has no prebuilt
wheel and has to be compiled; Python 3.12 and 3.13 install a ready wheel.

    py -3.13 -m venv .venv313
    .\.venv313\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install "gymnasium[box2d]" opencv-python-headless numpy
    pip install torch --index-url https://download.pytorch.org/whl/cu128   # or a CPU build

## Usage

    python train.py --total-steps 2000000
    python test_car_racing.py checkpoint_CarRacing-v3_best.pth 50

Main options: `--num-envs` (default 8), `--total-steps`, `--buffer-size` (default 300000,
about 2.1 GB RAM), `--updates-per-step`, `--seed`, `--no-render`.

