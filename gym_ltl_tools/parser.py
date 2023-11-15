import re
from typing import Pattern

import numpy as np

from gym_ltl_tools.typing import SymbolProp


class Parsers:
    def __init__(self):
        self.splitter: Pattern = re.compile(r"[\s]*(\d+|\w+|.)")

        self.parentheses: list[str] = ["(", ")"]

        self.symbols: dict[str, SymbolProp] = {
            "!": SymbolProp(3, lambda x: -x),
            "|": SymbolProp(1, lambda x, y: np.maximum(x, y)),
            "&": SymbolProp(2, lambda x, y: np.minimum(x, y)),
            "<": SymbolProp(3, lambda x, y: y - x),
            ">": SymbolProp(3, lambda x, y: x - y),
            "->": SymbolProp(3, lambda x, y: np.maximum(-x, y)),
            "<-": SymbolProp(3, lambda x, y: np.minimum(x, -y)),
            "F": SymbolProp(4, lambda x: np.max(x, axis=len(x.shape) - 1)),
            "G": SymbolProp(4, lambda x: np.min(x, axis=len(x.shape) - 1)),
        }


_parsers = Parsers()
