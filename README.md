# gym-tl-tools: Temporal Logic Wrappers for Gymnasium Environments

Utilities to wrap gymnasium environments using Temporal Logic (TL) rewards.

## Installation

You can install `gym-tl-tools` using pip:

```bash
pip install gym-tl-tools
```

Or, if you are developing locally, clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/gym-tl-tools.git
cd gym-tl-tools
pip install -e .
```

## Requirements
- Python 3.10+

## Usage

### 1. Define Atomic Predicates
Create a list of `Predicate` objects, each representing an atomic proposition in your TL formula. Make sure that your environment's `info` dictionary contains the necessary variables to evaluate these predicates.

```python
from gym_tl_tools import Predicate

atomic_predicates = [
    Predicate("goal_reached", "d_robot_goal < 1.0"),
    Predicate("obstacle_hit", "d_robot_obstacle < 1.0"),
]

# Ensure that the environment's info dictionary contains these variables.
# For example, info might look like: {"d_robot_goal": 3.0, "d_robot_obstacle": 1.0}
_, info = your_env.reset()
print(info)
# Output: {'d_robot_goal': 3.0, 'd_robot_obstacle': 1.0}
```

### 2. Specify the Temporal Logic Formula
Write your TL specification as a string, using the names of your atomic predicates.

```python
tl_spec = "F(goal_reached) & G(!obstacle_hit)"
```

### 3. Wrap Your Environment
Pass your environment, TL specification, and atomic predicates to `TlObservationReward`:

```python
from gym_tl_tools import TlObservationReward

wrapped_env = TlObservationReward(
    env,
    tl_spec=tl_spec,
    atomic_predicates=atomic_predicates,
)
```

### 4. Observation Structure
- If the original observation space is a `Dict`, the automaton state is added as a new key (default: `"aut_state"`).
- If the original observation space is a `Tuple`, the automaton state is appended.
- Otherwise, the observation is wrapped in a `Dict` with keys `"obs"` and `"aut_state"`.

### 5. Reward Calculation
At each step, the wrapper computes the reward based on the automaton's transition, reflecting progress toward (or violation of) the TL specification. The automaton state is updated according to the values of the atomic predicates, which are expected to be present in the `info` dictionary returned by the environment.

### 6. Reset and Step
On `reset()`, the automaton is reset to its initial state, and the initial observation is augmented. On `step(action)`, the automaton transitions based on the environment's `info`, and the reward is computed accordingly.

```python
obs, info = wrapped_env.reset()
done = False
while not done:
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    done = terminated or truncated
```

### Example
```python
import gymnasium as gym

from gym_tl_tools import Predicate
from gym_tl_tools import TlObservationReward

atomic_predicates = [
    Predicate("goal_reached", "d_robot_goal < 1.0"),
    Predicate("obstacle_hit", "d_robot_obstacle < 1.0"),
]
tl_spec = "F(goal_reached) & G(!obstacle_hit)"

# Ensure that the environment's info dictionary contains these variables.
env = gym.make("YourEnv-v0")  # Replace with your actual environment
# For example, info might look like: {"d_robot_goal": 3.0, "d_robot_obstacle": 1.0}
_, info = env.reset()
print(info)
# Output: {'d_robot_goal': 3.0, 'd_robot_obstacle': 1.0}

wrapped_env = TlObservationReward(
    env,
    tl_spec=tl_spec,
    atomic_predicates=atomic_predicates,
)

obs, info = wrapped_env.reset()
done = False
while not done:
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    done = terminated or truncated
```

## Citing gym-tl-tools
If you use this package in your research, please cite it as follows:

```bibtex
@misc{gym-tl-tools,
  author = {Mikihisa Yuasa},
  title = {gym-tl-tools: Temporal Logic Wrappers for Gymnasium Environments},
  year = {2025},
  howpublished = {\url{https://github.com/miki-yuasa/gym-tl-tools}},
  note = {Version 0.1.0}
}
```

## License
MIT License