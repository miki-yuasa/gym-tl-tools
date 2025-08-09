Examples
========

This page contains detailed examples showing how to use gym-tl-tools in different scenarios.

Basic Navigation Task
---------------------

Here's a complete example for a robot navigation task where the robot must reach a goal while avoiding obstacles:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from gym_tl_tools import (
       Predicate, 
       BaseVarValueInfoGenerator, 
       TLObservationReward,
       RewardConfig
   )

   # Define atomic predicates for navigation
   atomic_predicates = [
       Predicate(name="goal_reached", formula="d_robot_goal < 1.0"),
       Predicate(name="obstacle_hit", formula="d_robot_obstacle < 0.5"),
       Predicate(name="moving_forward", formula="forward_velocity > 0.1"),
   ]

   # Variable extraction from environment
   class NavigationVarGenerator(BaseVarValueInfoGenerator):
       def get_var_values(self, env, obs, info):
           # Extract relevant variables from observation and info
           return {
               "d_robot_goal": info.get("distance_to_goal", float('inf')),
               "d_robot_obstacle": info.get("distance_to_obstacle", float('inf')),
               "forward_velocity": obs.get("velocity", [0, 0])[0],
           }

   # Temporal logic specification
   tl_spec = "F(goal_reached) & G(!obstacle_hit) & G(moving_forward)"

   # Custom reward configuration
   reward_config = RewardConfig(
       terminal_state_reward=10.0,
       state_trans_reward_scale=50.0,
       dense_reward=True,
       dense_reward_scale=0.05
   )

   # Wrap environment
   env = gym.make("YourNavigationEnv-v0")  # Replace with actual env
   wrapped_env = TLObservationReward(
       env,
       tl_spec=tl_spec,
       atomic_predicates=atomic_predicates,
       var_value_info_generator=NavigationVarGenerator(),
       reward_config=reward_config,
   )

   # Training loop example
   obs, info = wrapped_env.reset()
   total_reward = 0
   steps = 0

   while steps < 1000:
       action = wrapped_env.action_space.sample()  # Replace with your policy
       obs, reward, terminated, truncated, info = wrapped_env.step(action)
       
       total_reward += reward
       steps += 1
       
       if terminated or truncated:
           print(f"Episode finished after {steps} steps")
           print(f"Total reward: {total_reward}")
           print(f"Success: {info.get('is_success', False)}")
           print(f"Failure: {info.get('is_failure', False)}")
           break

Multi-Objective Task
--------------------

Example with multiple objectives that must be achieved in sequence:

.. code-block:: python

   # Predicates for a multi-stage task
   atomic_predicates = [
       Predicate(name="pickup_item", formula="has_item > 0.5"),
       Predicate(name="deliver_item", formula="at_delivery_zone > 0.5"),
       Predicate(name="battery_charged", formula="battery_level > 0.3"),
   ]

   class MultiObjectiveVarGenerator(BaseVarValueInfoGenerator):
       def get_var_values(self, env, obs, info):
           return {
               "has_item": float(info.get("carrying_item", False)),
               "at_delivery_zone": float(info.get("in_delivery_zone", False)),
               "battery_level": obs.get("battery", 0.0),
           }

   # Sequential task: pickup item, then deliver it, while maintaining battery
   tl_spec = "F(pickup_item & F(deliver_item)) & G(battery_charged)"

Safe Exploration
----------------

Example emphasizing safety constraints during exploration:

.. code-block:: python

   # Safety-focused predicates
   atomic_predicates = [
       Predicate(name="goal_reached", formula="d_goal < 1.0"),
       Predicate(name="safe_from_cliff", formula="d_cliff > 2.0"),
       Predicate(name="safe_speed", formula="speed < 3.0"),
       Predicate(name="collision_free", formula="d_obstacle > 1.0"),
   ]

   class SafetyVarGenerator(BaseVarValueInfoGenerator):
       def get_var_values(self, env, obs, info):
           position = obs.get("position", [0, 0])
           velocity = obs.get("velocity", [0, 0])
           
           return {
               "d_goal": np.linalg.norm(
                   np.array(position) - np.array(info.get("goal_position", [0, 0]))
               ),
               "d_cliff": info.get("distance_to_cliff", float('inf')),
               "speed": np.linalg.norm(velocity),
               "d_obstacle": info.get("min_obstacle_distance", float('inf')),
           }

   # Reach goal while maintaining all safety constraints
   tl_spec = "F(goal_reached) & G(safe_from_cliff & safe_speed & collision_free)"

   # Use strict safety rewards
   reward_config = RewardConfig(
       terminal_state_reward=20.0,  # High reward for success
       state_trans_reward_scale=200.0,  # High penalty for safety violations
       dense_reward=False,  # Sparse rewards for clearer safety signals
   )

Custom Parser Example
---------------------

Using a custom parser with additional operators:

.. code-block:: python

   from gym_tl_tools import Parser, ParserSymbol

   # Create custom parser with additional operators
   custom_parser = Parser()
   
   # Add custom operator for "until" (U)
   custom_parser.symbols["U"] = ParserSymbol(
       priority=2, 
       func=lambda x, y: np.minimum(y, np.maximum(x, y))  # Simplified until
   )

   # Use custom parser in wrapper
   wrapped_env = TLObservationReward(
       env,
       tl_spec="safe_speed U goal_reached",  # Safe speed until goal is reached
       atomic_predicates=atomic_predicates,
       var_value_info_generator=var_generator,
       parser=custom_parser,
   )

Working with Different Observation Spaces
-----------------------------------------

Examples for different types of observation spaces:

.. code-block:: python

   # For Dict observation spaces
   class DictObsVarGenerator(BaseVarValueInfoGenerator):
       def get_var_values(self, env, obs, info):
           # obs is already a dict
           return {
               "robot_x": obs["robot"]["position"][0],
               "robot_y": obs["robot"]["position"][1],
               "target_distance": obs["sensors"]["target_distance"],
           }

   # For Box observation spaces
   class BoxObsVarGenerator(BaseVarValueInfoGenerator):
       def get_var_values(self, env, obs, info):
           # obs is a numpy array
           return {
               "position_x": obs[0],
               "position_y": obs[1],
               "velocity": np.linalg.norm(obs[2:4]),
               "sensor_reading": obs[4],
           }

Error Handling and Debugging
----------------------------

Example with proper error handling and debugging:

.. code-block:: python

   class DebugVarGenerator(BaseVarValueInfoGenerator):
       def get_var_values(self, env, obs, info):
           try:
               var_values = {
                   "d_goal": info["distance_to_goal"],
                   "d_obstacle": info["distance_to_obstacle"],
               }
               
               # Validate values
               for key, value in var_values.items():
                   if not isinstance(value, (int, float)):
                       raise ValueError(f"Variable {key} must be numeric, got {type(value)}")
                   if np.isnan(value) or np.isinf(value):
                       print(f"Warning: {key} has non-finite value {value}")
                       
               return var_values
               
           except KeyError as e:
               raise ValueError(f"Required key missing from info: {e}")
           except Exception as e:
               print(f"Error in variable extraction: {e}")
               # Return default values to prevent crash
               return {
                   "d_goal": float('inf'),
                   "d_obstacle": float('inf'),
               }

Integration with Stable-Baselines3
----------------------------------

Example showing how to use with reinforcement learning libraries:

.. code-block:: python

   from stable_baselines3 import PPO
   from stable_baselines3.common.env_util import make_vec_env

   # Create wrapped environment
   def make_tl_env():
       env = gym.make("YourEnv-v0")
       return TLObservationReward(
           env,
           tl_spec=tl_spec,
           atomic_predicates=atomic_predicates,
           var_value_info_generator=var_generator,
       )

   # Create vectorized environment
   vec_env = make_vec_env(make_tl_env, n_envs=4)

   # Train with PPO
   model = PPO("MlpPolicy", vec_env, verbose=1)
   model.learn(total_timesteps=100000)

   # Evaluate
   obs = vec_env.reset()
   for _ in range(1000):
       action, _states = model.predict(obs, deterministic=True)
       obs, reward, done, info = vec_env.step(action)
       if done:
           break
