import pytest

from gym_tl_tools.automaton import Automaton, Predicate


@pytest.mark.parametrize(
    "spec, atomic_predicates, target_is_trap_state",
    [
        (
            "F(psi_a)",
            [Predicate("psi_a", "d_a < 0.5")],
            [False, False],
        ),
        (
            "F(psi_a & psi_b)",
            [Predicate("psi_a", "d_a < 0.5"), Predicate("psi_b", "d_b < 0.5")],
            [False, False],
        ),
        (
            "F(psi_a | psi_b)",
            [Predicate("psi_a", "d_a < 0.5"), Predicate("psi_b", "d_b < 0.5")],
            [False, False],
        ),
        (
            "F(psi_a & !psi_b)",
            [Predicate("psi_a", "d_a < 0.5"), Predicate("psi_b", "d_b < 0.5")],
            [False, False],
        ),
        (
            "G(psi_a)",
            [Predicate("psi_a", "d_a < 0.5")],
            [False, True],
        ),
        (
            "F(psi_a) & G(psi_b)",
            [Predicate("psi_a", "d_a < 0.5"), Predicate("psi_b", "d_b < 0.5")],
            [False, False, True],
        ),
    ],
)
def test_trap_edges(
    spec: str, atomic_predicates: list[Predicate], target_is_trap_state: list[bool]
) -> None:
    aut = Automaton(spec, atomic_predicates)

    is_trap_state: list[bool] = [edge.is_trap_state for edge in aut.edges]

    assert tuple(is_trap_state) == tuple(target_is_trap_state), (
        f"Expected trap states {target_is_trap_state}, "
        f"but got {is_trap_state} for spec '{spec}'"
    )
