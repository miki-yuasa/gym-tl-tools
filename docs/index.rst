.. gym-tl-tools documentation master file, created by
   sphinx-quickstart on Sat Aug  9 12:00:00 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gym-tl-tools: Temporal Logic Wrappers for Gymnasium Environments
================================================================

Welcome to gym-tl-tools documentation! This package provides utilities to wrap Gymnasium environments using Temporal Logic (TL) rewards.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Installation
============

You can install ``gym-tl-tools`` using pip:

.. code-block:: bash

   pip install gym-tl-tools

Or, if you are developing locally, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/miki-yuasa/gym-tl-tools.git
   cd gym-tl-tools
   pip install -e .

Quick Start
===========

Here's a minimal example of how to use gym-tl-tools:

.. code-block:: python

   from gym_tl_tools import Predicate, BaseVarValueInfoGenerator, TLObservationReward
   import gymnasium as gym

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

   # Wrap your environment
   env = gym.make("YourEnv-v0")  # Replace with your actual environment
   wrapped_env = TLObservationReward(
       env,
       tl_spec="F(goal_reached) & G(!obstacle_hit)",
       atomic_predicates=atomic_predicates,
       var_value_info_generator=MyVarValueGenerator(),
   )

API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   gym_tl_tools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
