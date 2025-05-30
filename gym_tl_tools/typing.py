from typing import NamedTuple


class Transition(NamedTuple):
    """
    Represents a transition in a finite state automaton from a LTL formula.

    Attributes
    ----------
    next_state : int
        The next automaton state after the transition.
    condition : str
        The condition that triggers the transition represented as a Boolean expression.
        e.g. "psi_1 & psi_2" or "psi_1 | !psi_2".
    is_trapped_next : bool
        Indicates if the next state is a trap state (i.e., no further transitions are possible).
        Defaults to False.
    """

    condition: str
    next_state: int
    is_trapped_next: bool = False


class Predicate(NamedTuple):
    """
    Represents a predicate in a TL formula.

    Attributes
    ----------
    name : str
        The name of the atomic predicate.
        e.g. "psi_goal_robot".
    formula : str
        The formula of the atomic predicate, which can be a Boolean expression.
        e.g. "d_goal_robot < 5".
    """

    name: str
    formula: str
