from typing import Any, Callable, Generic, SupportsFloat, TypedDict

import numpy as np
from gymnasium import Env, ObservationWrapper
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Dict, Discrete, Tuple
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray
from pydantic import BaseModel

from gym_tl_tools.automaton import Automaton, Predicate
from gym_tl_tools.parser import Parser


class RewardConfigDict(TypedDict, total=False):
    """
    Configuration dictionary for the reward structure in the TLObservationReward wrapper.
    This dictionary is used to define how rewards are computed based on the automaton's state and the environment's info.

    Attributes
    ----------
    terminal_state_reward : float
        Reward given when the automaton reaches a terminal state in addition to the automaton state transition reward.
    state_trans_reward_scale : float
        Scale factor for the reward based on the automaton's state transition robustness.
        This is applied to the robustness computed from the automaton's transition.
        If the transition leads to a trap state, the reward is set to be negative, scaled by this factor.
    dense_reward : bool
        Whether to use dense rewards (True) or sparse rewards (False).
        Dense rewards provide a continuous reward signal based on the robustness of the transition to the next non-trap state.
    dense_reward_scale : float
        Scale factor for the dense reward.
        This is applied to the computed dense reward based on the robustness to the next non-trap automaton's state.
        If dense rewards are enabled, this factor scales the reward returned by the automaton.
    """

    terminal_state_reward: float
    state_trans_reward_scale: float
    dense_reward: bool
    dense_reward_scale: float


class RewardConfig(BaseModel):
    terminal_state_reward: float = 5
    state_trans_reward_scale: float = 100
    dense_reward: bool = False
    dense_reward_scale: float = 0.01


class TLObservationReward(
    ObservationWrapper[dict[str, ObsType | np.int64 | NDArray], ActType, ObsType],
    RecordConstructorArgs,
    Generic[ObsType, ActType],
):
    """
    A wrapper for Gymnasium environments that augments observations with the state of a temporal logic automaton,
    and computes rewards based on satisfaction of temporal logic (TL) specifications.

    This wrapper is designed for environments where the agent's objective is specified using temporal logic (e.g., LTL).
    It integrates an automaton (from a TL formula) into the observation and reward structure, enabling RL agents to
    learn tasks with complex temporal requirements.

    Usage
    -----
    1. **Define Atomic Predicates**:
        Create a list of `Predicate` objects, each representing an atomic proposition in your TL formula.
        Make sure that your `info` dictionary from the environment contains the necessary variables to evaluate these predicates.
        Example:
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

    2. **Specify the Temporal Logic Formula**:
        Write your TL specification as a string, using the names of your atomic predicates.
        Example:
            ```python
            tl_spec = "F(goal_reached) & G(!obstacle_hit)"
            ```

    3. **Wrap Your Environment**:
        Pass your environment, TL specification, and atomic predicates to [TLObservationReward](http://_vscodecontentref_/0).
        Example:
            ```python
            from gym_tl_tools import TLObservationReward
            wrapped_env = TLObservationReward(
                env,
                tl_spec=tl_spec,
                atomic_predicates=atomic_predicates,
            )
            ```

    4. **Observation Structure**:
        - The wrapper augments each observation with the current automaton state.
        - If the original observation space is a [Dict](https://gymnasium.farama.org/main/api/spaces/composite/#gymnasium.spaces.Dict), the automaton state is added as a new key (default: `"aut_state"`).
        - ~~If the original observation space is a [Tuple](https://gymnasium.farama.org/main/api/spaces/composite/#gymnasium.spaces.Tuple), the automaton state is appended.~~
        - Otherwise, the observation is wrapped in a [Dict](https://gymnasium.farama.org/main/api/spaces/composite/#gymnasium.spaces.Dict) with keys `"obs"` and `"aut_state"`.

    5. **Reward Calculation**:
        - At each step, the wrapper computes the reward based on the automaton's transition, which reflects progress toward (or violation of) the TL specification.
        - The automaton state is updated according to the values of the atomic predicates, which are expected to be present in the [info](http://_vscodecontentref_/4) dictionary returned by the environment.

    6. **Reset and Step**:
        - On `reset()`, the automaton is reset to its initial state, and the initial observation is augmented.
        - On `step(action)`, the automaton transitions based on the environment's [info](http://_vscodecontentref_/7), and the reward is computed accordingly.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap.
    tl_spec : str
        The temporal logic specification (e.g., LTL formula) to be used for the automaton.
    atomic_predicates : list[gym_tl_tools.Predicate]
        List of atomic predicates used in the TL formula.
    parser : Parser = gym_tl_tools.Parser()
        Parser for TL expressions (default: new instance of `Parser`.
    dict_aut_state_key : str = "aut_state"
        Key for the automaton state in the observation dictionary (default: "aut_state").

    Example
    -------
    ```python
    from gym_tl_tools import Predicate
    from gym_tl_tools import TLObservationReward

    atomic_predicates = [
        Predicate("goal_reached", "d_robot_goal < 1.0"),
        Predicate("obstacle_hit", "d_robot_obstacle < 1.0"),
    ]
    tl_spec = "F(goal_reached) & G(!obstacle_hit)"

    wrapped_env = TLObservationReward(
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
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        tl_spec: str,
        atomic_predicates: list[Predicate],
        var_value_info_generator: Callable[
            [Env[ObsType, ActType], ObsType, dict[str, Any]], dict[str, Any]
        ] = lambda env, obs, info: {},
        reward_config: RewardConfigDict = {
            "terminal_state_reward": 5,
            "state_trans_reward_scale": 100,
            "dense_reward": False,
            "dense_reward_scale": 0.01,
        },
        early_termination: bool = True,
        parser: Parser = Parser(),
        dict_aut_state_key: str = "aut_state",
    ):
        """
        Initialize the TLObservationReward wrapper.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The environment to wrap.
        tl_spec : str
            The temporal logic specification to be used with the automaton.
        atomic_predicates : list[gym_tl_tools.Predicate]
            A list of atomic predicates that define the conditions for the automaton.
            e.g. [Predicate("p1", lambda obs: obs["p1"] > 0.5)]
        parser : Parser = gym_tl_tools.Parser()
            An instance of the Parser class for parsing temporal logic expressions.
            Defaults to a new instance of Parser.
        var_value_info_generator : Callable[[Env[ObsType, ActType], ObsType, dict[str, Any]], dict[str, Any]] = lambda env, obs, info: {}
            A function that generates a dictionary of variable values from the environment's observation and info.
            This function should return a dictionary where keys are variable names and values are their corresponding values.
            This is used to evaluate the atomic predicates in the automaton.
        reward_config : RewardConfigDict = {
            "terminal_state_reward": 5,
            "state_trans_reward_scale": 100,
            "dense_reward": False,
            "dense_reward_scale": 0.01,
        }
            A dictionary containing the configuration for the reward structure.
            - `terminal_state_reward` : float
                Reward given when the automaton reaches a terminal state in addition to the automaton state transition reward.
            - `state_trans_reward_scale` : float
                Scale factor for the reward based on the automaton's state transition robustness.
                This is applied to the robustness computed from the automaton's transition.
                If the transition leads to a trap state, the reward is set to be negative, scaled by this factor.
            - `dense_reward` : bool
                Whether to use dense rewards (True) or sparse rewards (False).
                Dense rewards provide a continuous reward signal based on the robustness of the transition to the next non-trap state.
            - `dense_reward_scale` : float
                Scale factor for the dense reward.
                This is applied to the computed dense reward based on the robustness to the next non-trap automaton's state.
                If dense rewards are enabled, this factor scales the reward returned by the automaton.
        early_termination : bool = True
            Whether to terminate the episode when the automaton reaches a terminal state.
        dict_aut_state_key : str = "aut_state"
            The key under which the automaton state will be stored in the observation space.
            Defaults to "aut_state".
        """
        RecordConstructorArgs.__init__(
            self,
            tl_spec=tl_spec,
            atomic_predicates=atomic_predicates,
            parser=parser,
            reward_config=reward_config,
            dict_aut_state_key=dict_aut_state_key,
            early_termination=early_termination,
            var_value_info_generator=var_value_info_generator,
        )
        ObservationWrapper.__init__(self, env)
        self.var_value_info_generator = var_value_info_generator
        self.parser = Parser()
        self.automaton = Automaton(tl_spec, atomic_predicates, parser=parser)
        self.reward_config = RewardConfig(**reward_config)
        self.early_termination: bool = early_termination

        aut_state_space = Discrete(self.automaton.num_states)

        self._append_data_func: Callable[
            [ObsType, int], dict[str, ObsType | np.int64 | NDArray]
        ]
        # Find the observation space
        match type(env.observation_space):
            case Dict():
                assert dict_aut_state_key not in env.observation_space.spaces, (
                    f"Key '{dict_aut_state_key}' already exists in the observation space. "
                    "Please choose a different key."
                )
                observation_space = Dict(
                    {
                        **env.observation_space.spaces,
                        dict_aut_state_key: aut_state_space,
                    }
                )
                self._append_data_func = lambda obs, aut_state: {
                    **obs,
                    dict_aut_state_key: aut_state,
                }
            # case Tuple():
            #     observation_space = Tuple(
            #         env.observation_space.spaces + (aut_state_space,)
            #     )
            #     self._append_data_func = lambda obs, aut_state: obs + (aut_state,)
            case _:
                observation_space = Dict(
                    {"obs": env.observation_space, dict_aut_state_key: aut_state_space}
                )
                self._append_data_func = lambda obs, aut_state: {
                    "obs": obs,
                    "aut_state": np.int64(aut_state),
                }

        self.observation_space = observation_space
        self._obs_postprocess_func = lambda obs: obs

    def observation(
        self, observation: ObsType
    ) -> dict[str, ObsType | np.int64 | NDArray]:
        """
        Process the observation to include the automaton state.

        Parameters
        ----------
        observation : ObsType
            The original observation from the environment.

        Returns
        -------
        new_obs: dict[str,ObsType|int]
            The processed observation with the automaton state appended.
        """
        aut_state = self.automaton.current_state
        new_obs: dict[str, ObsType | np.int64 | NDArray] = self._append_data_func(
            observation, aut_state
        )
        return new_obs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, ObsType | np.int64 | NDArray], dict[str, Any]]:
        """
        Reset the environment and return the initial observation.

        Parameters
        ----------
        seed : int | None, optional
            Random seed for reproducibility.
        options : dict[str, Any] | None, optional
            Additional options for resetting the environment.

        Returns
        -------
        new_obs: dict[str,ObsType|int]
            The initial observation with the automaton state.
        info: dict[str, Any]
            Additional information from the reset.
            Should contain the variable keys and values that define the atomic predicates.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info_updates = self.var_value_info_generator(self.env, obs, info)
        # Update the info dict with the variable values
        info.update(info_updates)
        self.automaton.reset(seed=seed)
        new_obs = self.observation(obs)
        return new_obs, info

    def step(
        self, action: ActType
    ) -> tuple[
        dict[str, ObsType | np.int64 | NDArray],
        SupportsFloat,
        bool,
        bool,
        dict[str, Any],
    ]:
        """
        Take a step in the environment with the given action.

        Parameters
        ----------
        action : ActType
            The action to take in the environment.

        Returns
        -------
        new_obs: dict[str,ObsType|int]
            The new observation after taking the action.
        reward: SupportsFloat
            The reward received from the environment.
        terminated: bool
            Whether the episode has terminated.
        truncated: bool
            Whether the episode has been truncated.
        info: dict[str, Any]
            Additional information from the step.
            Should contain the variable keys and values that define the atomic predicates.
        """
        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        info_updates = self.var_value_info_generator(self.env, obs, info)
        # Update the info dict with the variable values
        info.update(info_updates)
        info.update({"original_reward": orig_reward})
        reward, next_aut_state = self.automaton.step(
            info, **self.reward_config.model_dump()
        )
        if (
            self.early_termination
            and next_aut_state
            in self.automaton.goal_states + self.automaton.trap_states
        ):
            terminated = True

        if next_aut_state in self.automaton.goal_states:
            info.update(
                {
                    "is_success": True,
                }
            )
        else:
            info.update(
                {
                    "is_success": False,
                }
            )

        new_obs = self.observation(obs)
        return new_obs, reward, terminated, truncated, info
