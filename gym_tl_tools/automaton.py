import math
import random
import re
from collections import deque
from typing import Literal, NamedTuple

import spot
from spot import twa as Twa

from gym_tl_tools.parser import Parser
from gym_tl_tools.typing import Predicate, Transition


class RobustnessCounter(NamedTuple):
    robustness: float
    ind: int


class AutomatonStateCounter(NamedTuple):
    state: int
    ind: int


class Edge:
    """
    Represents an edge in a finite state automaton from a LTL formula.
    Attributes
    ----------
    state : int
        The automaton state.
    transitions : list[Transition]
        The transitions from this state to other states.
    is_inf_zero_acc : bool
        Indicates if the edge has an Inf(0) acceptance condition.
    is_terminal_state : bool
        Indicates if the edge is a terminal state (i.e., no further transitions are possible).
    is_trap_state : bool
        Indicates if the edge is a trap state (i.e., all transitions lead to trap states).
    """

    def __init__(self, raw_state: str, atomic_pred_names: list[str]):
        """
        Parameters
        ----------
        raw_state : str
            The raw state string from the automaton in HOA format.
        atomic_pred_names : list[str]
            The names of the atomic predicates used in the transitions.
        """
        self.state: int = int(*re.findall("State: (\d).*\[", raw_state))
        self.transitions: list[Transition] = [
            Transition(
                replace_atomic_pred_ids_to_names(
                    add_or_parentheses(re.findall("\[(.*)\]", transition)[0]),
                    atomic_pred_names,
                ),
                int(*re.findall("\[.*\] (\d)", transition)),
            )
            for transition in re.findall(
                "(\[.*\] \d+)\n", raw_state.replace("[", "\n[")
            )
        ]
        # Check if the edge has an Inf(0) acceptance condition
        self.is_inf_zero_acc = True if "{0}" in raw_state else False
        # Check if the edge is a terminal state. If so, elimite 't' transition
        # looping back to itself
        is_terminal_state: bool = True
        for i, transition in enumerate(self.transitions):
            if transition.next_state != self.state:
                is_terminal_state = False
            else:
                if transition.condition == "t":
                    self.transitions.pop(i)
                else:
                    pass
        self.is_terminal_state = is_terminal_state
        # A terminal state is a trap state if it doesn't have Inf(0) acceptance
        self.is_trap_state: bool = (
            True if not self.is_inf_zero_acc and self.is_terminal_state else False
        )


class Automaton:
    def __init__(
        self,
        tl_spec: str,
        atomic_predicates: list[Predicate],
        parser: Parser = Parser(),
    ) -> None:
        self.tl_spec: str = tl_spec
        self.atomic_predicates: list[Predicate] = atomic_predicates
        self.parser: Parser = parser

        aut: Twa = spot.translate(tl_spec, "Buchi", "state-based", "complete")
        aut_hoa: str = aut.to_str("hoa")
        self.num_states: int = int(aut.num_states())
        self.start: int = int(*re.findall("Start: (\d+)\n", aut_hoa))

        num_used_aps: int = int(*re.findall('AP: (\d+) "', aut_hoa))
        used_aps: list[str] = (
            re.findall("AP: \d+ (.*)", aut_hoa)[0].replace('"', "").split()
        )
        if len(used_aps) != num_used_aps:
            raise ValueError(
                f"Number of used atomic predicates ({len(used_aps)}) does not match the number of atomic predicates in the automaton ({num_used_aps})."
            )
        else:
            self.used_aps: list[str] = used_aps

        self.acc_name: str = re.findall("acc-name: (.*)\n", aut_hoa)[0]
        self.acceptance: str = re.findall("Acceptance: (.*)\n", aut_hoa)[0]
        self.properties = re.findall("properties: (.*)\n", aut_hoa)

        aut_hoa_states = (
            aut_hoa.replace("\n", "")
            .replace("State", "\nState")
            .replace("--END--", "\n")
        )
        raw_states: list[str] = [
            raw_state + "\n" for raw_state in re.findall("(State:.*)\n", aut_hoa_states)
        ]

        untrapped_edges: list[Edge] = [
            Edge(raw_state, self.used_aps) for raw_state in raw_states
        ]

        # Check if a transition of an edge leads to a trap state and if all transitions
        # of an edge leads to a trap state (if so, this edge is also a trap state)
        trap_states: list[int] = [
            edge.state for edge in untrapped_edges if edge.is_trap_state
        ]
        goal_states: list[int] = []
        for _ in range(len(untrapped_edges)):
            for i, edge in enumerate(untrapped_edges):
                if edge.is_trap_state:
                    pass
                else:
                    is_trapped_in_all_trans = True
                    for j, transition in enumerate(edge.transitions):
                        if transition.next_state in trap_states:
                            if edge.is_inf_zero_acc:
                                edge.transitions.pop(j)
                                goal_states.append(edge.state)
                            else:
                                edge.transitions[j] = Transition(
                                    transition.condition, transition.next_state, True
                                )
                        else:
                            is_trapped_in_all_trans = False

                    if is_trapped_in_all_trans:
                        untrapped_edges[i].is_trap_state = True
                        trap_states.append(edge.state)
                    else:
                        pass

        self.goal_states: tuple[int, ...] = tuple(goal_states)
        self.trap_states: tuple[int, ...] = tuple(trap_states)
        self.edges: tuple[Edge, ...] = tuple(untrapped_edges)

    def reset(self) -> None:
        """
        Reset the automaton to its initial state.
        """
        self.current_state: int = self.start
        self.status: Literal["intermediate", "goal", "trap"] = "intermediate"

    def step(self, var_value_dict: dict[str, float]) -> tuple[float, int]:
        """
        Step the automaton to the next state based on the current predicate values.
        This function updates the current state of the automaton based on the
        values of variables defining atomic predicates provided in `var_value_dict`.

        Parameters
        ----------
        var_value_dict : dict[str, float]
            A dictionary mapping the variable names used in the atomic predicate definitions
            to their current values.
            The keys should match the names of the atomic predicates

        Returns
        -------
        reward : float
            The reward for the current step.
        next_state : int
            The next automaton state.
        """

        if not hasattr(self, "current_state"):
            raise ValueError(
                " Error: The automaton has not been reset. Please call the reset() method before stepping."
            )

        ap_rob_dict: dict[str, float] = {
            atom_pred.name: self.parser.tl2rob(atom_pred.formula, var_value_dict)
            for atom_pred in self.atomic_predicates
        }

        reward, next_state = self.tl_reward(
            ap_rob_dict, self.current_state, dense_reward=False
        )

        # Update the current state of the automaton
        self.current_state = next_state

        # Update the status of the automaton
        if next_state in self.goal_states:
            self.status = "goal"
        elif next_state in self.trap_states:
            self.status = "trap"
        else:
            self.status = "intermediate"

        return reward, next_state

    def tl_reward(
        self,
        ap_rob_dict: dict[str, float],
        curr_aut_state: int,
        dense_reward: bool = False,
        terminal_state_reward: float = 5,
    ) -> tuple[float, int]:
        """
        Calculate the reward of the step from a given automaton.

        Parameters
        ----------
        ap_rob_dict: dict[str, float]
            A dictionary mapping the names of atomic predicates to their robustness values.
        curr_aut_state: int
            The current state of the automaton.
        dense_reward: bool
            If True, the reward is calculated based on the maximum robustness of non-trap transitions.
        terminal_state_reward: float
            The reward given when the automaton reaches a terminal state.

        Returns
        -------
        reward : float
            The reward of the step based on the MDP and automaton states.
        next_aut_state : int
            The resultant automaton state after the step.
        """

        if curr_aut_state in self.goal_states:
            return (terminal_state_reward, curr_aut_state)
        elif curr_aut_state in self.trap_states:
            return (-terminal_state_reward, curr_aut_state)
        else:
            curr_edge = self.edges[curr_aut_state]
            transitions = curr_edge.transitions

            # Calculate robustnesses of the transitions
            robs, non_trap_robs, trap_robs = self.transition_robustness(
                transitions, ap_rob_dict
            )

            positive_robs: list[RobustnessCounter] = [
                RobustnessCounter(rob, i) for i, rob in enumerate(robs) if rob > 0
            ]

            # Check if there is only one positive transition robustness unless there are
            # multiple 0's
            if len(positive_robs) != 1:
                is_all_positive_zero: bool = all(
                    int(pos_rob.robustness) == 0 for pos_rob in positive_robs
                )
                if is_all_positive_zero:
                    is_containing_trap_state: bool = False
                    trap_index = 0
                    for i, pos_rob in enumerate(positive_robs):
                        if transitions[pos_rob.ind].is_trapped_next:
                            is_containing_trap_state = True
                            trap_index = i
                            break
                        else:
                            pass

                    if is_containing_trap_state:
                        positive_robs = [positive_robs[trap_index]]
                    else:
                        next_states: list[AutomatonStateCounter] = []
                        for pos_rob_ind, pos_rob in enumerate(positive_robs):
                            next_states.append(
                                AutomatonStateCounter(
                                    transitions[pos_rob.ind].next_state, pos_rob_ind
                                )
                            )
                        next_state_inds: list[int] = [
                            state.ind
                            for state in next_states
                            if state.ind != curr_aut_state
                        ]
                        if next_state_inds:
                            positive_robs = [
                                positive_robs[random.choice(next_state_inds)]
                            ]
                        else:  # should only contain the current state as the next state
                            positive_robs = [random.choice(positive_robs)]

                else:
                    raise ValueError(
                        "Error: Only one of the transition robustnesses can be positive.",
                        "The positive transitions were:",
                        [transitions[rob.ind].condition for rob in positive_robs],
                    )
            else:
                pass
            positive_rob: RobustnessCounter = deque(positive_robs).pop()
            trans_rob: float = positive_rob.robustness
            next_aut_state: int = transitions[positive_rob.ind].next_state

            # Calculate the reward
            reward: float
            # Weight for reward calculation
            alpha: float = 0.7
            beta: float = 0.5
            gamma: float = 0.01
            delta: float = 100

            if next_aut_state == curr_aut_state:
                # non_trap_robs.remove(trans_rob)
                # reward = -gamma * (
                #    beta * 1 / max(non_trap_robs) - (1 - beta) * 1 / max(trap_robs)
                # )

                # reward = gamma * (
                #    beta * 1 / max(non_trap_robs) - (1 - beta) * 1 / max(trap_robs)
                # )
                if dense_reward:
                    non_trap_robs.remove(trans_rob)
                    reward = gamma * max(non_trap_robs)
                else:
                    reward = 0
            else:
                if trans_rob in non_trap_robs:
                    reward = (
                        delta * trans_rob
                    )  # alpha * trans_rob - (1 - alpha) * max(trap_robs)
                elif trans_rob in trap_robs:
                    reward = (
                        -delta * trans_rob
                    )  # -(1 - alpha) * max(non_trap_robs) - alpha * trans_rob
                else:
                    raise ValueError(
                        "Error: the transition robustness doesn't exit in the robustness set."
                    )

            return (reward, next_aut_state)

    def transition_robustness(
        self,
        transitions: list[Transition],
        ap_rob_dict: dict[str, float],
    ) -> tuple[list[float], list[float], list[float]]:
        robs: list[float] = []
        non_trap_robs: list[float] = []
        trap_robs: list[float] = []
        for trans in transitions:
            rob: float = self.parser.tl2rob(trans.condition, ap_rob_dict)
            robs.append(rob)
            if trans.is_trapped_next:
                trap_robs.append(rob)
            else:
                non_trap_robs.append(rob)

        return robs, non_trap_robs, trap_robs


def add_or_parentheses(condition: str) -> str:
    or_locs: list[int] = [m.start() for m in re.finditer("\|", condition)]

    condition_list: list[str] = list(condition)

    new_condition: str = condition

    if or_locs:
        for loc in or_locs:
            if condition_list[loc - 1] == " " and condition_list[loc + 1] == " ":
                condition_list[loc - 1] = ")"
                condition_list[loc + 1] = "("
            else:
                pass
        new_condition = "(" + "".join(condition_list) + ")"
    else:
        pass

    return new_condition


def replace_atomic_pred_ids_to_names(condition: str, atom_prop_names: list[str]):
    spec: str = condition
    # Replace atomic props in numeric IDs to their actual names (ex. '0'->'psi_0')
    for i, atom_prop_name in enumerate(reversed(atom_prop_names)):
        target = str(len(atom_prop_names) - 1 - i)
        spec = spec.replace(target, atom_prop_name)

    return spec
