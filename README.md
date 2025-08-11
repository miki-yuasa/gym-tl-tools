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
uv sync
```

## Requirements
- Python 3.10+

## Usage

### 1. Define Atomic Predicates
Create a list of `Predicate` objects, each representing an atomic proposition in your TL formula. Each predicate has a `name` (used in the TL formula) and a `formula` (Boolean expression string for evaluation).

```python
from gym_tl_tools import Predicate

atomic_predicates = [
    Predicate(name="goal_reached", formula="d_robot_goal < 1.0"),
    Predicate(name="obstacle_hit", formula="d_robot_obstacle < 1.0"),
]
```

### 2. Create a Variable Value Generator
Implement a subclass of `BaseVarValueInfoGenerator` to extract variable values from the environment's observation and info for evaluating the atomic predicates.

```python
from gym_tl_tools import BaseVarValueInfoGenerator

class MyVarValueGenerator(BaseVarValueInfoGenerator):
    def get_var_values(self, env, obs, info):
        # Extract variables needed for predicate evaluation
        return {
            "d_robot_goal": info.get("d_robot_goal", float('inf')),
            "d_robot_obstacle": info.get("d_robot_obstacle", float('inf')),
        }

var_generator = MyVarValueGenerator()
```

### 2. Create a Variable Value Generator
Implement a subclass of `BaseVarValueInfoGenerator` to extract variable values from the environment's observation and info for evaluating the atomic predicates.

```python
from gym_tl_tools import BaseVarValueInfoGenerator

class MyVarValueGenerator(BaseVarValueInfoGenerator):
    def get_var_values(self, env, obs, info):
        # Extract variables needed for predicate evaluation
        return {
            "d_robot_goal": info.get("d_robot_goal", float('inf')),
            "d_robot_obstacle": info.get("d_robot_obstacle", float('inf')),
        }

var_generator = MyVarValueGenerator()
```

### 3. Specify the Temporal Logic Formula
Write your TL specification as a string, using the names of your atomic predicates.

```python
tl_spec = "F(goal_reached) & G(!obstacle_hit)"
```

### 4. Wrap Your Environment
Pass your environment, TL specification, atomic predicates, and variable value generator to `TLObservationReward`:

```python
from gym_tl_tools import TLObservationReward

wrapped_env = TLObservationReward(
    env,
    tl_spec=tl_spec,
    atomic_predicates=atomic_predicates,
    var_value_info_generator=var_generator,
)
```

### 5. Observation Structure
- If the original observation space is a `Dict`, the automaton state is added as a new key (default: `"aut_state"`).
- If the original observation space is a `Tuple`, the automaton state is appended.
- Otherwise, the observation is wrapped in a `Dict` with keys `"obs"` and `"aut_state"`.

### 6. Reward Calculation
At each step, the wrapper computes the reward based on the automaton's transition, reflecting progress toward (or violation of) the TL specification. The automaton state is updated according to the values of the atomic predicates, which are evaluated using the variable values provided by the `var_value_info_generator`.

### 7. Reset and Step
On `reset()`, the automaton is reset to its initial state, and the initial observation is augmented. On `step(action)`, the automaton transitions based on the variable values extracted by the `var_value_info_generator`, and the reward is computed accordingly.

```python
obs, info = wrapped_env.reset()
done = False
while not done:
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    done = terminated or truncated
```

### Notes
- The wrapper adds additional information to the `info` dict, including:
    - `is_success`: Whether the automaton has reached a goal state.
    - `is_failure`: Whether the automaton has reached a trap state.
    - `is_aut_terminated`: Whether the automaton has been terminated.

### Example
```python
import gymnasium as gym

from gym_tl_tools import Predicate, BaseVarValueInfoGenerator, TLObservationReward

# Define atomic predicates
atomic_predicates = [
    Predicate(name="goal_reached", formula="d_robot_goal < 1.0"),
    Predicate(name="obstacle_hit", formula="d_robot_obstacle < 1.0"),
]

# Create variable value generator
class MyVarValueGenerator(BaseVarValueInfoGenerator):
    def get_var_values(self, env, obs, info):
        return {
            "d_robot_goal": info.get("d_robot_goal", float('inf')),
            "d_robot_obstacle": info.get("d_robot_obstacle", float('inf')),
        }

var_generator = MyVarValueGenerator()
tl_spec = "F(goal_reached) & G(!obstacle_hit)"

# Ensure that the environment's info dictionary contains these variables.
env = gym.make("YourEnv-v0")  # Replace with your actual environment
# For example, info might look like: {"d_robot_goal": 3.0, "d_robot_obstacle": 1.0}
_, info = env.reset()
print(info)
# Output: {'d_robot_goal': 3.0, 'd_robot_obstacle': 1.0}

wrapped_env = TLObservationReward(
    env,
    tl_spec=tl_spec,
    atomic_predicates=atomic_predicates,
    var_value_info_generator=var_generator,
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

## Documentation

Full documentation is available at: https://miki-yuasa.github.io/gym-tl-tools/

### Building Documentation Locally

To build the documentation locally:

```bash
# Install documentation dependencies
uv sync --group dev

# Build documentation
./build_docs.sh

# Or manually:
cd docs
uv run sphinx-build -b html . _build/html
```

The built documentation will be available at `docs/_build/html/index.html`.