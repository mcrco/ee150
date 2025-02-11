from graphviz import Digraph
from grad import Value
from grad.lib.viz_lib import draw_dot

# ------------------------------------------------------------ initialize values
w1: Value = Value(1, name="w1")
w2: Value = Value(2.3, name="w2")  # TODO: (2.i) name this Value "w2"
w3: Value = Value(5.7, name="w3")  # TODO: (2.i) name this Value "w3"

# ------------------------------------------------------------- build expression
s = w1 + w2
s.name = "s"
L = w3 * s
L.name = "L"

# backpropogate to calculate all gradients
# hint: call backward on L
L.backward()

# -------------------------------------------- render computation graph (before)
dot: Digraph = draw_dot(L, show_grad=True)
dot.render("figs/ex_backprop_before", format="png", view=True, cleanup=True)

# ------------------------------------------------------------------ update step

# update all parameters with eta=0.01
eta = 0.01
w1.data = w1.data - eta * w1.grad  # TODO: (2.i) write code to update params
w2.data = w2.data - eta * w2.grad  # TODO: (2.i) write code to update params
w3.data = w3.data - eta * w3.grad  # TODO: (2.i) write code to update params

# -------------------------------------------- render computation graph (after)
s_after = w1 + w2
s_after.name = "s"
L_after = w3 * s_after
L_after.name = "L"

dot: Digraph = draw_dot(L_after, show_grad=False)
dot.render("figs/ex_backprop_after", format="png", view=True, cleanup=True)
