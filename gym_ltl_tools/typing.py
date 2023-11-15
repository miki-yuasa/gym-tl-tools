from typing import Callable, NamedTuple, TypeAlias, Literal, TypedDict, Any

from numpy.typing import NDArray


class SymbolProp(NamedTuple):
    priority: int
    func: Callable[..., float | int | NDArray | Any]


class VarProp(NamedTuple):
    name: str
    args: list[str]
    func: Callable[..., float | int | NDArray | Any]
