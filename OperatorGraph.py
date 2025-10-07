from __future__ import annotations
from typing import Any, Dict, List, Tuple, Set, Optional, Callable
import math


class Operator:
    """
    Base operator node.
    - name: unique name used for graph ident.
    - inputs: names of input nodes (strings referencing other Operator.name)
    - value: computed forward value
    - grad: gradient w.r.t. output (for backward)
    """

    def __init__(self, name: str, inputs: Optional[List[str]] = None):
        self.name = name
        self.inputs = inputs or []
        self.value: Any = None
        self.grad: Optional[float] = None

    def forward(self, input_values: List[Any]) -> Any:
        """Compute forward pass given evaluated inputs (in same order as self.inputs)."""
        raise NotImplementedError

    def backward(self, input_values: List[Any], upstream_grad: float) -> List[float]:
        """
        Compute gradients to propagate to inputs.
        Returns list of gradients in same order as input_values.
        By default, raise if no backward defined.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class Input(Operator):
    """Holds an externally provided value."""

    def __init__(self, name: str):
        super().__init__(name, inputs=[])
        # Input value will be set externally in graph.execute()

    def forward(self, input_values: List[Any]) -> Any:
        return self.value

    def backward(self, input_values: List[Any], upstream_grad: float) -> List[float]:
        # Input nodes don't propagate further
        return []


class Add(Operator):
    """Elementwise add. Accepts two inputs."""

    def __init__(self, name: str, a: str, b: str):
        super().__init__(name, inputs=[a, b])

    def forward(self, input_values: List[float]) -> float:
        return input_values[0] + input_values[1]

    def backward(self, input_values: List[float], upstream_grad: float) -> List[float]:
        # d(a+b)/da = 1, d(a+b)/db = 1
        return [upstream_grad, upstream_grad]


class Mul(Operator):
    """Multiply two scalar inputs."""

    def __init__(self, name: str, a: str, b: str):
        super().__init__(name, inputs=[a, b])

    def forward(self, input_values: List[float]) -> float:
        return input_values[0] * input_values[1]

    def backward(self, input_values: List[float], upstream_grad: float) -> List[float]:
        a, b = input_values
        # d(a*b)/da = b, d(a*b)/db = a
        return [upstream_grad * b, upstream_grad * a]


class Linear(Operator):
    """Simple linear operator y = w*x + b.
    Expects inputs: [x, w, b] (scalars) — small toy example.
    """

    def __init__(self, name: str, x: str, w: str, b: str):
        super().__init__(name, inputs=[x, w, b])

    def forward(self, input_values: List[float]) -> float:
        x, w, b = input_values
        return w * x + b

    def backward(self, input_values: List[float], upstream_grad: float) -> List[float]:
        x, w, b = input_values
        # dy/dx = w, dy/dw = x, dy/db = 1
        return [upstream_grad * w, upstream_grad * x, upstream_grad * 1.0]


class ReLU(Operator):
    def __init__(self, name: str, x: str):
        super().__init__(name, inputs=[x])

    def forward(self, input_values: List[float]) -> float:
        (x,) = input_values
        return x if x > 0 else 0.0

    def backward(self, input_values: List[float], upstream_grad: float) -> List[float]:
        (x,) = input_values
        local_grad = 1.0 if x > 0 else 0.0
        return [upstream_grad * local_grad]


class OperatorGraph:
    """
    Holds operators by name. Connects them by operator.inputs references.
    Provides:
        - add_operator(op)
        - execute(inputs: mapping input_name->value) -> mapping name->value
        - backward(target_name, loss_grad=1.0) -> mapping param_name->grad
    """

    def __init__(self):
        self.nodes: Dict[str, Operator] = {}

    def add_operator(self, op: Operator):
        if op.name in self.nodes:
            raise KeyError(f"Operator with name '{op.name}' already exists.")
        self.nodes[op.name] = op

    def _topological_order(self, outputs: Optional[List[str]] = None) -> List[str]:
        """
        Returns a topological ordering of nodes needed to compute 'outputs'.
        If outputs is None, returns order for entire graph.
        Simple DFS-based topo sorted order.
        """
        visited: Set[str] = set()
        temp: Set[str] = set()
        order: List[str] = []

        def visit(n: str):
            if n in visited:
                return
            if n in temp:
                raise ValueError("Graph contains a cycle (not a DAG).")
            temp.add(n)
            node = self.nodes.get(n)
            if node is None:
                raise KeyError(f"Node '{n}' not found in graph.")
            for inp in node.inputs:
                visit(inp)
            temp.remove(n)
            visited.add(n)
            order.append(n)

        if outputs is None:
            outputs = list(self.nodes.keys())
        for out in outputs:
            visit(out)
        return order

    def execute(self, input_values: Dict[str, Any], outputs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Forward execute all nodes needed for 'outputs' (or all nodes if outputs None).
        input_values: mapping from Input node names to their scalar values.
        Returns mapping of node.name -> computed value for all executed nodes.
        """
        # set Input node values
        for name, val in input_values.items():
            node = self.nodes.get(name)
            if node is None:
                raise KeyError(f"Input node '{name}' not found.")
            if not isinstance(node, Input):
                raise TypeError(f"Node '{name}' is not an Input node.")
            node.value = val

        order = self._topological_order(outputs)
        # forward pass
        for name in order:
            node = self.nodes[name]
            if isinstance(node, Input):
                # already has value
                continue
            # gather inputs' values
            input_vals = [self.nodes[inp].value for inp in node.inputs]
            node.value = node.forward(input_vals)
        # return values for nodes in order (or entire dict)
        return {name: self.nodes[name].value for name in order}

    def backward(self, loss_node_name: str, loss_grad: float = 1.0) -> Dict[str, float]:
        """
        Simple backward propagation of scalars through the DAG.
        Assumes forward has been called and node.value are set.
        Returns mapping node_name -> gradient of loss wrt node output.
        """
        # init grads to 0
        grads: Dict[str, float] = {n: 0.0 for n in self.nodes}
        grads[loss_node_name] = loss_grad

        # process nodes in reverse topo order of all nodes needed to compute loss_node
        order = self._topological_order([loss_node_name])
        for name in reversed(order):
            node = self.nodes[name]
            upstream = grads.get(name, 0.0)
            node.grad = upstream
            if upstream == 0.0:
                continue
            # compute input values
            if not node.inputs:
                continue  # Input nodes or 0-input ops
            input_vals = [self.nodes[inp].value for inp in node.inputs]
            try:
                input_grads = node.backward(input_vals, upstream)
            except NotImplementedError:
                raise NotImplementedError(f"Backward not implemented for node {name}")
            # accumulate gradients to inputs
            for inp_name, g in zip(node.inputs, input_grads):
                grads[inp_name] = grads.get(inp_name, 0.0) + float(g)
        return grads

    def to_dot(self, outputs: Optional[List[str]] = None) -> str:
        """Export a DOT graph string for visualization with Graphviz."""
        order = self._topological_order(outputs)
        lines = ["digraph OperatorGraph {"]
        for name in order:
            op = self.nodes[name]
            label = f"{name}\\n{op.__class__.__name__}"
            lines.append(f'  "{name}" [label="{label}"];')
            for inp in op.inputs:
                lines.append(f'  "{inp}" -> "{name}";')
        lines.append("}")
        return "\n".join(lines)


# -------------------------
# Example usage / toy test
# Build a small graph computing:
#   z = ReLU( Linear(x, w, b) + c )
# where c is an extra bias added after linear.
# -------------------------
if __name__ == "__main__":
    g = OperatorGraph()

    # Inputs
    g.add_operator(Input("x"))
    g.add_operator(Input("w"))
    g.add_operator(Input("b"))
    g.add_operator(Input("c"))

    # Linear: t = w*x + b
    g.add_operator(Linear("t", x="x", w="w", b="b"))
    # Add: u = t + c
    g.add_operator(Add("u", a="t", b="c"))
    # Activation: z = ReLU(u)
    g.add_operator(ReLU("z", x="u"))

    # Forward run
    inputs = {"x": 2.0, "w": 3.0, "b": 0.5, "c": -1.0}
    values = g.execute(inputs, outputs=["z"])
    print("Forward values:", values)  # z should be ReLU(3*2 + 0.5 -1) = ReLU(5.5) = 5.5

    # Backward run: dL/dz = 1.0 (assume loss equals z)
    grads = g.backward("z", loss_grad=1.0)
    print("Gradients (dL/dnode):")
    for k in ["z", "u", "t", "x", "w", "b", "c"]:
        print(f"  {k}: {grads.get(k, 0.0)}")

    # DOT output for visualization
    print("\nDOT format (Graphviz):")
    print(g.to_dot(["z"]))
