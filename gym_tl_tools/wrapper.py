from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, SupportsFloat

import numpy as np
from gymnasium import Env, ObservationWrapper
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Dict, Discrete
from gymnasium.utils import RecordConstructorArgs
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializationInfo,
    model_serializer,
)
from typing_extensions import TypedDict

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


class BaseVarValueInfoGenerator(Generic[ObsType, ActType], ABC):
    """
    Base class for generating variable values from the environment's observation and info.

    This class should be subclassed to implement custom logic for extracting variable values
    that are used to evaluate atomic predicates in temporal logic formulas.

    The `get_var_values` method should return a dictionary where keys are variable names
    used in predicate formulas (e.g., "d_robot_goal", "d_robot_obstacle") and values are
    their corresponding numerical values extracted from the environment's observation and info.

    Example
    -------
    ```python
    class MyVarValueGenerator(BaseVarValueInfoGenerator):
        def get_var_values(self, env, obs, info):
            return {
                "d_robot_goal": info.get("d_robot_goal", float('inf')),
                "d_robot_obstacle": obs.get("obstacle_distance", 0.0),
                "robot_speed": np.linalg.norm(obs["velocity"]) if "velocity" in obs else 0.0,
            }
    ```
    """

    @abstractmethod
    def get_var_values(
        self, env: Env[ObsType, ActType], obs: ObsType, info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract variable values from the environment's observation and info.

        This method should be implemented by subclasses to provide the variable values
        needed for evaluating atomic predicates in temporal logic formulas.

        Parameters
        ----------
        env : Env[ObsType, ActType]
            The environment from which to extract variable values.
        obs : ObsType
            The current observation from the environment.
        info : dict[str, Any]
            The info dictionary containing additional information from the environment.

        Returns
        -------
        dict[str, Any]
            A dictionary where keys are variable names used in predicate formulas
            and values are their corresponding numerical values.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class TLObservationRewardConfig(BaseModel, Generic[ObsType, ActType]):
    """
    Configuration for the TLObservationReward wrapper.

    This class defines the parameters used to configure the TLObservationReward wrapper,
    including the temporal logic specification, atomic predicates, variable value generator,
    reward configuration, and other settings.

    Attributes
    ----------
    tl_spec : str
        The temporal logic specification (e.g., LTL formula) to be used for the automaton.
    atomic_predicates : list[Predicate]
        List of atomic predicates used in the TL formula.
    var_value_info_generator : BaseVarValueInfoGenerator[ObsType, ActType]
        An instance of a subclass of BaseVarValueInfoGenerator that extracts variable values from the environment's observation and info.
    reward_config : RewardConfigDict
        Configuration for the reward structure.
    early_termination : bool
        Whether to terminate episodes when automaton reaches terminal states.
    parser : Parser
        Parser for TL expressions.
    dict_aut_state_key : str
        Key for the automaton state in the observation dictionary.
    """

    tl_spec: str = ""
    atomic_predicates: list[Predicate]
    var_value_info_generator: BaseVarValueInfoGenerator[ObsType, ActType] | None = None
    reward_config: RewardConfig = RewardConfig()
    early_termination: bool = True
    parser: Parser = Parser()
    dict_aut_state_key: str = "aut_state"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_serializer
    def serialize_model(self, info: SerializationInfo | None = None) -> dict[str, Any]:
        """Custom model serializer to handle atomic_predicates serialization with context."""
        # Handle atomic_predicates based on context
        atomic_predicates_data = self.atomic_predicates
        # if info and info.context:
        #     excluded = info.context.get("excluded", [])
        #     if "atomic_predicates" not in excluded:
        #         atomic_predicates_data = [
        #             pred.model_dump() for pred in self.atomic_predicates
        #         ]
        # else:
        #     atomic_predicates_data = [
        #         pred.model_dump() for pred in self.atomic_predicates
        #     ]

        data = {
            "tl_spec": self.tl_spec,
            "atomic_predicates": atomic_predicates_data,
            "var_value_info_generator": self.var_value_info_generator,
            "reward_config": self.reward_config,
            "early_termination": self.early_termination,
            "parser": self.parser,
            "dict_aut_state_key": self.dict_aut_state_key,
        }
        return data


class TLObservationReward(
    ObservationWrapper[
        dict[str, ObsType | np.int64 | NDArray[np.number]], ActType, ObsType
    ],
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
        Each predicate has a `name` (used in the TL formula) and a `formula` (Boolean expression string).
        Example:
            ```python
            from gym_tl_tools import Predicate
            atomic_predicates = [
                Predicate(name="goal_reached", formula="d_robot_goal < 1.0"),
                Predicate(name="obstacle_hit", formula="d_robot_obstacle < 1.0"),
            ]
            ```

    2. **Create a Variable Value Generator**:
        Implement a subclass of `BaseVarValueInfoGenerator` to extract variable values from the environment's
        observation and info for evaluating the atomic predicates.
        Example:
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

    3. **Specify the Temporal Logic Formula**:
        Write your TL specification as a string, using the names of your atomic predicates.
        Example:
            ```python
            tl_spec = "F(goal_reached) & G(!obstacle_hit)"
            ```

    4. **Wrap Your Environment**:
        Pass your environment, TL specification, atomic predicates, and variable value generator to `TLObservationReward`.
        Example:
            ```python
            from gym_tl_tools import TLObservationReward
            wrapped_env = TLObservationReward(
                env,
                tl_spec=tl_spec,
                atomic_predicates=atomic_predicates,
                var_value_info_generator=var_generator,
            )
            ```

    5. **Observation Structure**:
        - The wrapper augments each observation with the current automaton state.
        - If the original observation space is a [Dict](https://gymnasium.farama.org/main/api/spaces/composite/#gymnasium.spaces.Dict), the automaton state is added as a new key (default: `"aut_state"`).
        - ~~If the original observation space is a [Tuple](https://gymnasium.farama.org/main/api/spaces/composite/#gymnasium.spaces.Tuple), the automaton state is appended.~~
        - Otherwise, the observation is wrapped in a [Dict](https://gymnasium.farama.org/main/api/spaces/composite/#gymnasium.spaces.Dict) with keys `"obs"` and `"aut_state"`.

    6. **Reward Calculation**:
        - At each step, the wrapper computes the reward based on the automaton's transition, which reflects progress toward (or violation of) the TL specification.
        - The automaton state is updated according to the values of the atomic predicates, which are evaluated using the variable values provided by the `var_value_info_generator`.

    7. **Reset and Step**:
        - On `reset()`, the automaton is reset to its initial state, and the initial observation is augmented.
        - On `step(action)`, the automaton transitions based on the variable values extracted by the `var_value_info_generator`, and the reward is computed accordingly.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap.
    tl_spec : str
        The temporal logic specification (e.g., LTL formula) to be used for the automaton.
    atomic_predicates : list[gym_tl_tools.Predicate]
        List of atomic predicates used in the TL formula. Each predicate has a `name` (used in the TL formula)
        and a `formula` (Boolean expression string for evaluation).
    var_value_info_generator : BaseVarValueInfoGenerator[ObsType, ActType]
        An instance of a subclass of `BaseVarValueInfoGenerator` that extracts variable values from the
        environment's observation and info for evaluating atomic predicates.
    reward_config : RewardConfigDict, optional
        Configuration for the reward structure (default values provided).
    early_termination : bool, optional
        Whether to terminate episodes when automaton reaches terminal states (default: True).
    parser : Parser, optional
        Parser for TL expressions (default: new instance of `Parser`).
    dict_aut_state_key : str, optional
        Key for the automaton state in the observation dictionary (default: "aut_state").

    Example
    -------
    ```python
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
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        tl_spec: str,
        atomic_predicates: list[Predicate],
        var_value_info_generator: BaseVarValueInfoGenerator[ObsType, ActType],
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
            Each predicate should have a `name` (used in the TL formula) and a `formula` (Boolean expression string).
            e.g. [Predicate(name="goal_reached", formula="d_robot_goal < 1.0")]
        var_value_info_generator : BaseVarValueInfoGenerator[ObsType, ActType]
            An instance of a subclass of BaseVarValueInfoGenerator that extracts variable values from the
            environment's observation and info. This is used to evaluate the atomic predicates in the automaton.
            The get_var_values method should return a dictionary where keys are variable names used in predicate
            formulas and values are their corresponding numerical values.
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

        match tl_spec:
            case "0" | "1" | "":
                raise ValueError(
                    f"The temporal logic specification cannot be '0', '1', or an empty string (given tl_spec={tl_spec}). "
                    "Please provide a valid temporal logic formula."
                )

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
        self.var_value_info_generator: BaseVarValueInfoGenerator[ObsType, ActType] = (
            var_value_info_generator
        )
        self.parser = Parser()
        self.automaton = Automaton(tl_spec, atomic_predicates, parser=parser)
        self.reward_config = RewardConfig(**reward_config)
        self.early_termination: bool = early_termination

        aut_state_space = Discrete(self.automaton.num_states)

        self._append_data_func: Callable[
            [ObsType, int], dict[str, ObsType | np.int64 | NDArray]
        ]
        # Find the observation space
        match env.observation_space:
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
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = {
            "forward_aut_on_reset": True,
            "obs": {},
            "info": {},
        },
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
        # Update the info dict with the variable values
        info = self._var_value_info_update(obs, info)
        self.automaton.reset(seed=seed)

        if options:
            if options.get("forward_aut_on_reset", True):
                obs = options["obs"] if options.get("obs", {}) else obs
                info = options["info"] if options.get("info", {}) else info
                self.forward_aut(obs, info)

        new_obs = self.observation(obs)
        return new_obs, info

    def forward_aut(self, obs: ObsType, info: dict[str, Any]) -> None:
        """
        Forward the automaton based on the current observation and info.

        This method updates the automaton state based on the current observation and info.
        It is useful for manually stepping the automaton without taking an action in the environment.

        Parameters
        ----------
        obs : ObsType
            The current observation from the environment.
        info : dict[str, Any]
            The info dictionary containing variable values for the atomic predicates.
        """
        curr_state: int = self.automaton.current_state
        new_state: int | None = None
        # Update the automaton state based on the info
        info = self._var_value_info_update(obs, info)
        while new_state != curr_state:
            curr_state = self.automaton.current_state
            _, new_state = self.automaton.step(info, **self.reward_config.model_dump())
        # Update the automaton state
        if new_state is not None:
            self.automaton.current_state = new_state

    @property
    def is_aut_terminated(self) -> bool:
        """
        Check if the automaton is in a terminal state.

        Returns
        -------
        bool
            True if the automaton is in a terminal state, False otherwise.
        """
        return self.automaton.is_terminated

    def _var_value_info_update(
        self, obs: ObsType, info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Update the info dictionary with variable values based on the observation.

        This method is used to update the info dictionary with the values of the atomic predicates
        based on the current observation. It is called during the step and reset methods.

        Parameters
        ----------
        obs : ObsType
            The current observation from the environment.
        info : dict[str, Any]
            The info dictionary to be updated with variable values.

        Returns
        -------
        info : dict[str, Any]
            The updated info dictionary containing the variable keys and values that define the atomic predicates.
        """
        info_updates = self.var_value_info_generator.get_var_values(self.env, obs, info)
        info.update(info_updates)

        return info

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
        info = self._var_value_info_update(obs, info)
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

        info.update(
            {
                "is_success": next_aut_state in self.automaton.goal_states,
                "is_aut_terminated": next_aut_state
                in self.automaton.goal_states + self.automaton.trap_states,
            }
        )

        new_obs = self.observation(obs)
        return new_obs, reward, terminated, truncated, info
