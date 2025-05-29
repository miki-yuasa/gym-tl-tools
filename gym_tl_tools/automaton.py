import glob
import os
import re

import spot
from spot import twa

from gym_tl_tools.typing import AtomicPredicates, Transition


class Edge:
    def __init__(self, raw_state: str, atomic_pred_names: list[str]):
        self.state: int = int(*re.findall("State: (\d).*\[", raw_state))
        self.transitions: list[Transition] = [
            Transition(
                replace_atomic_pred_digits_to_name(
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


class TLAutomaton:
    def __init__(self, tl_spec: str) -> None:
        self.tl_spec: str = tl_spec
        aut = spot.translate(tl_spec, "Buchi", "state-based", "complete")
        aut_hoa = aut.to_str("hoa")
        self.num_states: int = int(aut.num_states())
        self.start: int = int(*re.findall("Start: (\d+)\n", aut_hoa))
        self.ap: AtomicPredicates = AtomicPredicates(
            int(*re.findall('AP: (\d+) "', aut_hoa)),
            re.findall("AP: \d+ (.*)", aut_hoa)[0].replace('"', "").split(),
        )
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
            Edge(raw_state, self.ap.aps) for raw_state in raw_states
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

        self.goal_states = goal_states
        self.trap_states = trap_states
        self.edges = untrapped_edges


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


def replace_atomic_pred_digits_to_name(condition: str, atom_prop_names: list[str]):
    spec: str = condition
    # Replace atomic props in digits to their acctual names (ex. '0'->'psi_0')
    for i, atom_prop_name in enumerate(reversed(atom_prop_names)):
        target = str(len(atom_prop_names) - 1 - i)
        spec = spec.replace(target, atom_prop_name)

    return spec
