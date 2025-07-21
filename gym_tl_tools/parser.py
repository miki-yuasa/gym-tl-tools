import re
from collections import deque
from typing import Any, Callable, Pattern, cast

import numpy as np
from pydantic import BaseModel


class ParserSymbol(BaseModel):
    """
    Represents a symbol in the parser with its priority and function.

    Attributes
    ----------
    priority : int
        The priority of the symbol, used for parsing precedence.
    func : Callable[..., Any]
        The function that implements the operation of the symbol.
        It should take the appropriate number of arguments and return a value.
    """

    priority: int
    func: Callable[..., Any]


class Parser:
    def __init__(self):
        self.splitter: Pattern = re.compile(r"[\s]*(\d+|\w+|.)")

        self.parentheses: list[str] = ["(", ")"]

        self.symbols: dict[str, ParserSymbol] = {
            "!": ParserSymbol(priority=3, func=lambda x: -x),
            "|": ParserSymbol(priority=1, func=lambda x, y: np.maximum(x, y)),
            "&": ParserSymbol(priority=2, func=lambda x, y: np.minimum(x, y)),
            "<": ParserSymbol(priority=3, func=lambda x, y: y - x),
            ">": ParserSymbol(priority=3, func=lambda x, y: x - y),
            "->": ParserSymbol(priority=3, func=lambda x, y: np.maximum(-x, y)),
            "<-": ParserSymbol(priority=3, func=lambda x, y: np.minimum(x, -y)),
            "F": ParserSymbol(
                priority=4, func=lambda x: np.max(x, axis=len(x.shape) - 1)
            ),
            "G": ParserSymbol(
                priority=4, func=lambda x: np.min(x, axis=len(x.shape) - 1)
            ),
        }

    # Check if a token is a parenthesis
    def is_parentheses(self, s: str, **kwargs) -> bool:
        if "index" in kwargs:
            return s is self.parentheses[kwargs["index"]]
        else:
            return s in self.parentheses

    # Check if a token is symbol
    def is_symbol(self, s: Any) -> bool:
        return s in self.symbols

    # Check if a token is number
    def is_num(self, s: str) -> bool:
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True

    # Check if a token is variable
    def is_var(self, s: str, var_names: list[str]) -> bool:
        return s in var_names

    # Priorities of Symbol
    def get_priority(self, s: str) -> int:
        return self.symbols[s].priority

    # Funcion of Symbol
    def get_func(self, s: str) -> Callable[..., Any]:
        return self.symbols[s].func

    # Tokenize the spec
    def tokenize(self, spec: str) -> list[str]:
        token_list_tmp = deque(self.splitter.findall(spec))
        token_list: list[str] = []
        while token_list_tmp:
            token = token_list_tmp.popleft()
            if token == ".":
                if self.is_num(token_list[-1]) and self.is_num(token_list_tmp[0]):
                    token_list.append(
                        token_list.pop() + token + token_list_tmp.popleft()
                    )
                else:
                    raise ValueError(
                        "Error: invalid '.' in the spec. It should be used between two numbers."
                    )
            elif token == "-":
                if token_list[-1] == "<":
                    token_list.append(token_list.pop() + token)
                elif token_list_tmp[0] == ">":
                    token_list.append(token + token_list_tmp.popleft())
                else:
                    raise ValueError(
                        "Error: invalid '-' in the spec. It should be used between two numbers or symbols."
                    )
            else:
                token_list.append(token)
        return token_list

    def parse(self, token_list: list[str], var_names: list[str]) -> list[str]:
        """
        Convert the token to Reverse Polish Notation
        """
        stack: list[str] = []
        output_stack: list[str] = []

        for i in range(len(token_list)):
            token: str = token_list.pop(0)
            # If a number, push to the output stack
            if self.is_num(token) | self.is_var(token, var_names):
                output_stack.append(token)
            # If a starting parenthesis, push to stack
            elif self.is_parentheses(token, index=0):
                stack.append(token)
            # If an ending parenthesis, pop and add to the
            # output stack until the starting parenthesis
            # comes.
            elif self.is_parentheses(token, index=1):
                for i in range(len(stack)):
                    symbol: str = stack.pop()
                    if self.is_parentheses(symbol, index=0):
                        break
                    output_stack.append(symbol)
            # If the read token's priority is less than that
            # of the one in the end of the stack, pop from
            # the stack, add to the output stack, and then
            # add to the stack
            elif (
                stack
                and self.is_symbol(stack[-1])
                and self.get_priority(token) <= self.get_priority(stack[-1])
            ):
                symbol = stack.pop()
                output_stack.append(symbol)
                stack.append(token)
            # Push the others to the stack
            else:
                stack.append(token)
        # Finally, add the stack to the ouput stack
        while stack:
            output_stack.append(stack.pop(-1))
        return output_stack

    def evaluate(
        self,
        parsed_tokens: list[str],
        var_dict: dict[str, float],
    ) -> float:
        """
        Checking from the start, get tokens from the stack and
        compute there if there is a symbol
        """
        output: list[str | float] = [token for token in parsed_tokens]
        cnt: int = 0
        while len(output) != 1:
            if self.is_symbol(output[cnt]):
                symbol: str = cast(str, output.pop(cnt))
                num_args: int = self.symbols[symbol].func.__code__.co_argcount
                target_index: int = cnt - num_args
                args: list[float] = []
                for i in range(num_args):
                    arg: str = output.pop(target_index)
                    if self.is_num(arg):
                        args.append(float(arg))
                    elif arg.isascii():
                        args.append(cast(float, var_dict[arg]))
                    else:
                        raise ValueError(
                            "Error: the token should be either a number or a variable."
                        )
                result = self.get_func(symbol)(*args)
                output.insert(target_index, str(result))
                cnt = target_index + 1
            else:
                cnt += 1

        final_result: float = (
            float(var_dict[output[0]])
            if isinstance(output[0], str) and output[0] in var_dict.keys()
            else float(output[0])
        )

        return final_result

    def tl2rob(self, spec: str, var_dict: dict[str, float]) -> float:
        """
        Parsing a TL spec to its robustness

        Parameters
        ----------
        spec : str
            The TL spec to be parsed.
            e.g. "psi_1 & psi_2 | !psi_3".
        var_dict : dict[str, float]
            A dictionary mapping the variable names used in the TL spec to their current values.
            The keys should match the names of the atomic predicates defined in the spec.
            e.g. {"d_robot_goal": 3.0, "d_robot_obstacle": 1.0, "d_robot_goal": 0.5}.

        Returns
        -------
        robustness : float
            The robustness value of the TL spec given the variable values.
            A positive value indicates satisfaction, while a negative value indicates violation.
        """

        token_list: list[str] = self.tokenize(spec)
        parsed_tokens: list[str] = self.parse(token_list, list(var_dict.keys()))
        robustness: float = self.evaluate(parsed_tokens, var_dict)

        return robustness


def replace_special_characters(spec: str) -> str:
    """
    Replace special characters in the spec with underscores so that it can be used as a file name.

    Parameters
    ----------
    spec: str
        spec to be replaced.
        The spec can contain special characters like spaces, &, |, etc.
        These characters are replaced with underscores to create a valid file name.

    Returns
    -------
    replaced_spec: str
        spec with special characters replaced by underscores

    Examples
    --------
    ```python
    spec = replace_special_characters("F(psi_1 | psi_2) & G(!psi_3)")
    print(spec)
    # Output: "F(psi_1_or_psi_2)_and_G(!psi_3)"
    ```
    """

    replaced_spec: str = (
        spec.replace(" ", "_")
        .replace("&", "_and_")
        .replace("|", "_or_")
        .replace("__", "_")
    )

    return replaced_spec
