Quick Start Guide
================

This guide will walk you through the basic usage of gym-tl-tools to wrap your Gymnasium environments with temporal logic rewards.

Basic Concepts
--------------

**Temporal Logic (TL)**: A formal specification language that allows you to express properties about sequences of states over time.

**Atomic Predicates**: Basic propositions that can be evaluated as true or false based on the current state.

**Automaton**: A finite state machine derived from your temporal logic formula that tracks progress toward goal satisfaction.

Step-by-Step Usage
------------------

1. Define Atomic Predicates
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create predicates that represent the basic conditions in your environment:

.. code-block:: python

   from gym_tl_tools import Predicate

   atomic_predicates = [
       Predicate(name="goal_reached", formula="d_robot_goal < 1.0"),
       Predicate(name="obstacle_hit", formula="d_robot_obstacle < 1.0"),
       Predicate(name="safe_speed", formula="robot_speed < 2.0"),
   ]

2. Create a Variable Value Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement a class that extracts variables from your environment's observations and info:

.. code-block:: python

   from gym_tl_tools import BaseVarValueInfoGenerator

   class MyVarValueGenerator(BaseVarValueInfoGenerator):
       def get_var_values(self, env, obs, info):
           return {
               "d_robot_goal": info.get("d_robot_goal", float('inf')),
               "d_robot_obstacle": info.get("d_robot_obstacle", float('inf')),
               "robot_speed": np.linalg.norm(obs.get("velocity", [0, 0])),
           }

3. Specify Your Temporal Logic Formula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write your specification using the predicate names:

.. code-block:: python

   # Eventually reach goal while always avoiding obstacles and maintaining safe speed
   tl_spec = "F(goal_reached) & G(!obstacle_hit & safe_speed)"

4. Wrap Your Environment
~~~~~~~~~~~~~~~~~~~~~~~~

Apply the wrapper to your Gymnasium environment:

.. code-block:: python

   from gym_tl_tools import TLObservationReward
   import gymnasium as gym

   env = gym.make("YourEnv-v0")
   wrapped_env = TLObservationReward(
       env,
       tl_spec=tl_spec,
       atomic_predicates=atomic_predicates,
       var_value_info_generator=MyVarValueGenerator(),
   )

5. Use the Wrapped Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapped environment can be used like any Gymnasium environment:

.. code-block:: python

   obs, info = wrapped_env.reset()
   print(f"Initial observation keys: {obs.keys()}")
   # Output: dict_keys(['obs', 'aut_state'])

   done = False
   total_reward = 0
   while not done:
       action = wrapped_env.action_space.sample()
       obs, reward, terminated, truncated, info = wrapped_env.step(action)
       total_reward += reward
       done = terminated or truncated
       
       # Check automaton status
       if info.get("is_success"):
           print("Goal achieved!")
       elif info.get("is_failure"):
           print("Specification violated!")

Understanding the Output
------------------------

**Observation Structure**:
- Original observations are preserved under the `"obs"` key
- Automaton state is added under the `"aut_state"` key (configurable)

**Reward Structure**:
- Rewards are based on automaton transitions and robustness values
- Positive rewards indicate progress toward goal satisfaction
- Negative rewards indicate violations or movement toward failure states

**Info Dictionary Additions**:
- `is_success`: Boolean indicating if the automaton reached a goal state
- `is_failure`: Boolean indicating if the automaton reached a trap state  
- `is_aut_terminated`: Boolean indicating if the automaton episode has ended

Temporal Logic Operators
-------------------------

The parser supports the following operators:

- `&`: Logical AND (conjunction)
- `|`: Logical OR (disjunction)  
- `!`: Logical NOT (negation)
- `F`: Eventually (future)
- `G`: Always (globally)
- `->`: Implication
- `<-`: Reverse implication

Example formulas:

.. code-block:: python

   # Eventually reach goal
   "F(goal_reached)"
   
   # Always avoid obstacles
   "G(!obstacle_hit)"
   
   # Eventually reach goal while always avoiding obstacles
   "F(goal_reached) & G(!obstacle_hit)"
   
   # If close to obstacle, then slow down
   "G(close_to_obstacle -> slow_speed)"
