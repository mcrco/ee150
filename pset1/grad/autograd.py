import math
from typing import Iterable, Union


class Value:
    """Value wrapper for composable autograd unit values."""

    def __init__(
        self,
        data: float,
        prev: Iterable = (),
        op: str = "",
        name: str = "",
    ) -> None:
        """Initialize a `Value` with number (int or float).

        Note: Properties prefixed with an underscore _ are internal properties,
        not to be touched by the user

        Parameters
        ----------
        data : float
            unit value (must be a float number)
        prev : tuple, optional
            Set of the previous Value's that made this Value, by default ()
        op : str, optional
            The operation that produced this Value (for debugging purposes),
            by default ""
        name : str, optional
            Give this Value a symbol name like "x" or "y" (debugging purposes).
            Symbol shows up when generating graph diagrams.
        """
        self.data: float = data
        # self.grad holds the value for (dL/dv) where v is this Value
        # by default, we set it to 0
        self.grad = 0.0
        # by default we set a dummy _backward function
        self._backward = lambda: None
        # by default, a node has no previous nodes unless provided
        self.prev = set(prev)
        self.op = op
        self.name = name

    @staticmethod
    def parse(x: Union[int, float, "Value"]) -> "Value":
        # NOTE: don't change this or your tests will fail!
        if isinstance(x, Value):
            return x
        assert isinstance(x, (int, float)), "Values can only hold float scalars"
        return Value(float(x))

    def __add__(self, other) -> "Value":
        # other should be a Value, so if it is int/float, parse it into a new Value
        other = Value.parse(other)

        res = Value(self.data + other.data, prev=(self, other), op="+")

        local_grad_self = 1.0  # dw_next/dw
        local_grad_other = 1.0  # dw_next/dw

        def _backward():
            self.grad += res.grad * local_grad_self  # dL/dw_next * dw_next/dw
            other.grad += res.grad * local_grad_other  # dL/dw_next * dw_next/dw

        res._backward = _backward
        return res

    def __radd__(self, other) -> "Value":
        # this will call __add__ as implemented above
        return self + other

    def __mul__(self, other) -> "Value":
        other = Value.parse(other)
        res = Value(self.data * other.data, prev=(self, other), op="×")

        local_grad_self = other.data
        local_grad_other = self.data

        def _backward():
            self.grad += res.grad * local_grad_self
            other.grad += res.grad * local_grad_other

        res._backward = _backward
        return res

    def __rmul__(self, other) -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other) -> "Value":
        return self + (-other)

    def __rsub__(self, other) -> "Value":
        return Value.parse(other) - self

    def __pow__(self, other) -> "Value":
        if not isinstance(other, (int, float)):
            raise NotImplementedError(
                "Value type can only be raised to int/float powers"
            )

        res = Value(self.data**other, prev=(self,), op="^")

        local_grad_self = other * (self.data ** (other - 1))

        def _backward():
            self.grad += res.grad * local_grad_self

        res._backward = _backward
        return res

    def __truediv__(self, other) -> "Value":
        other = Value.parse(other)
        if other.data == 0:
            raise ZeroDivisionError("Division by zero detected")

        return self * (other ** (-1))

    def __rtruediv__(self, other) -> "Value":
        return Value.parse(other) / self

    def f_relu(self) -> "Value":
        res_val = 0 if self.data < 0 else self.data
        res = Value(res_val, prev=(self,), op="fn_relu")

        local_grad_self = 0 if self.data < 0 else 1

        def _backward():
            self.grad += res.grad * local_grad_self

        res._backward = _backward
        return res

    def f_sigmoid(self) -> "Value":
        res_val = 1 / (1 + math.exp(-self.data))
        res = Value(res_val, prev=(self,), op="fn_σ")

        local_grad_self = res_val * (1 - res_val)

        def _backward():
            self.grad += res.grad * local_grad_self

        res._backward = _backward
        return res

    def f_tanh(self) -> "Value":
        res_val = math.tanh(self.data)
        res = Value(res_val, prev=(self,), op="fn_tanh")

        local_grad_self = 1 - res_val**2

        def _backward():
            self.grad += res.grad * local_grad_self

        res._backward = _backward
        return res

    def zero_grad(self) -> None:
        self.grad = 0.0

    def backward(self):
        from .lib.topo import (
            topo_sort,
        )  # import here to avoid circular import (don't change this please)

        vals_in_order = topo_sort(self)

        self.grad = 1
        for value in reversed(vals_in_order):
            value._backward()

    def __repr__(self) -> str:
        # NOTE: don't change this, or your tests will fail!
        if self.name == "":
            return f"Value({self.data:.3f}, grad={self.grad:.3f})"
        else:
            return f"Value({self.name}, {self.data:.3f}, grad={self.grad:.3f})"
