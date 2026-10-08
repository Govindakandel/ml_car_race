# gcoder — Car Agent trained with SARSA

A small reinforcement learning project: a car learns to drive down a road full of
obstacles using the **SARSA** algorithm. Rendered with `pygame`.

## Environment

`env.py` defines `CarWorld`, a top-down driving world:

- **Road**: the car starts at the top of a vertical road (600×700 window) and
  always moves forward; only the horizontal position changes with the action.
- **Obstacles**: 10 red rectangles scattered on the road.
- **Car** (green): moves left / straight / right, and forward every step.

### State space

The car has 3 sensors (left, straight, right), each returning whether an
obstacle is `CLOSE` (0) or `FAR` (1) within its detection range:

```
state = [left, straight, right]   # each ∈ {0, 1}  →  2³ = 8 states
```

Each state is mapped to an id with `state_id = left*4 + straight*2 + right`.

### Action space

| Action | Meaning  |
|--------|----------|
| 0      | LEFT     |
| 1      | STRAIGHT |
| 2      | RIGHT    |

### Rewards

| Event                    | Reward |
|--------------------------|--------|
| Crash into an obstacle   | -10    |
| Reach the end of the road| +100   |
| Every forward step       | +1     |

## Algorithm — SARSA

[SARSA](https://en.wikipedia.org/wiki/Sarsa) is an on-policy temporal-difference
algorithm. The Q-table is updated with the rule:

```
Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
```

where `s,a` are the current state/action, `r` the reward, and `s',a'` the
**next state and the action actually taken next** (sampled with the same
ε-greedy policy — that's what makes it on-policy).

Hyperparameters used (`sarsa.py`):

- `alpha` (learning rate) = 0.1
- `epsilon` (exploration rate) = 0.1
- `gamma` (discount factor) = 0.9
- episodes = 1000

### Q-table before training

```
[[ 0.   0.   0. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]
 [ 0.   0.   0. ]]
```

### Q-table after 1000 episodes of SARSA

```
[[12.84 10.24  5.89]
 [-1.03 11.23 15.52]
 [14.33 16.29 14.57]
 [15.26 14.94 17.18]
 [15.19 14.81  9.23]
 [ 8.33  8.7  11.59]
 [15.58 15.81 15.84]
 [27.54 19.42 18.84]]
```

(columns = LEFT, STRAIGHT, RIGHT)

## Trained agent rendering

After training, the greedy policy (no exploration) drives the car to the end of
the road:

https://github.com/Govindakandel/ml_car_race/gcoder/sarsa_agent.mp4

> Replace the URL above with your actual GitHub repo raw URL once pushed, or
> embed the video with an HTML tag (renders on GitHub):
>
> ```html
> <video src="sarsa_agent.mp4" controls></video>
> ```

## How to run

```bash
pip install pygame-ce numpy gymnasium
python sarsa.py
```

- `env.py` — the CarWorld environment (random agent demo: `python env.py`)
- `sarsa.py` — training loop + renders the trained agent
- `record.py` — records the trained agent to `sarsa_agent.mp4`

## TODO

- [ ] Q-learning (`Qlearning.py`) — coming soon
