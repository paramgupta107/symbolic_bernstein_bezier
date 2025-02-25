# Symbolic Bernstein Bezier

**symbolic_bernstein_bezier** is a Python library for performing operations on expressions in Bernstein–Bézier form in an arbitrary number of dimensions and degrees.

It uses [sympy](https://www.sympy.org/) to represent the Bernstein–Bézier coefficients.  All operations are only supported between instances of  `BernsteinBezier`. It supports conversion to and from sympy expressions. To perform additional operations, convert to sympy expression, perform operation and covert back to `BernsteinBezier`

## Table of Contents

  

-  [Usage](#usage)

-  [License](#license)

-  [Contact](#contact)

## Usage

### Create BernsteinBezier Object

Initialize a Bernstein–Bézier object by providing a list  of coefficients in row‐major order, a list of degrees (one per variable/dimension), and a list of sympy symbols for the variables.

```python
from symbolic_bernstein_bezier import BernsteinBezier
import sympy

# Define symbolic variables.
u, v = sympy.symbols("u v")

# Example for a 2D surface with degrees [1, 1] (i.e. 2x2 control points)
coeffs = [sympy.Symbol("c00"), sympy.Symbol("c01"),
          sympy.Symbol("c10"), sympy.Symbol("c11")]
degrees = [1, 1]
vars = [u, v]

# Create the BernsteinBezier instance.
bb = BernsteinBezier(coeffs, degrees, vars)
```

### Convert  BB to Sympy Expression

Convert a Bernstein–Bézier representation to a sympy expression.

```python
expr = bb.to_expr()
```

### Convert Sympy Expression to BB

Convert a sympy expression to a Bernstein–Bézier representation. It will fail if the sympy expression cannot be converted to a polynomial

```python
# Given an expression in t1 and t2.
expr = u**2 + 2*u*v + v**2
vars = [u, v]
bb_from_expr = BernsteinBezier.from_expr(expr, vars)
```

### Get Coefficients of BB

Retrieve individual control points or get all control points as a flat list in row major order.

```python
# Get a specific control point (e.g., at index (0, 1)).
cp = bb.get_coeff((0, 1))
print("Control point at (0,1):", cp)

# Get all control points as a flat list.
all_cps = bb.get_all_coeffs()
print("All control points:", all_cps)
```

### Arithmetic Operations (`__add__`, `__sub__`, `__mul__`)

Perform addition, subtraction, and multiplication of Bernstein–Bézier objects. If the degrees do not match, the degree at each dimension will be raised to the maximum degree of the two at that particular dimension.

```python
# Addition.
bb_sum = bb + bb

# Subtraction.
bb_diff = bb - bb

# Multiplication.
bb_prod = bb * bb
```

### Differentiate

Differentiate the Bernstein–Bézier representation with respect to a specified variable (dimension).

```python
# Differentiate with respect to the first variable (dimension 0).
bb_diff_u = bb.differentiate(0)
```

### Integrate

Integrate the Bernstein–Bézier representation with respect to a specified variable. Returns an instance of `BernsteinBezier` with one less dimension. If only one dimension remains after integration, a sympy expression is returned.

```python
# Integrate with respect to the second variable (dimension 1).
bb_int_v = bb.integrate(1)
```
### Extract Boundary

Extract the boundary of the Bernstein–Bézier representation along a specified dimension and index. Use -1 for the last index.

```python
# Extract the boundary along dimension 0 at index -1.
bb_boundary = bb.extract_boundary(0, -1)
```

### Compose

Compose the Bernstein–Bézier representation with other Bernstein–Bézier objects or sympy expressions. Required a list of `BernsteinBezier` or sympy expressions that represent the substitution for each variable.

```python 
g1 = BernsteinBezier.from_expr(s**2, [s]) 
g2 = BernsteinBezier.from_expr(s + t, [s, t]) 
bb_composed1 = bb.compose([g1, g2], [s, t]) 
print(bb_composed1) 

bb_composed2 = bb.compose([s**3, 2*t - 1], [s, t]) 
print(bb_composed2)
```

### Degree Raise

Raise the degree of the Bernstein–Bézier representation in each dimension.

```python
# Raise the degree by 1 in first dimension.
bb_raised = bb.degree_raise([1, 0])
```

### Subdivide

Subdivide the Bernstein–Bézier representation along a given dimension using De Casteljau's algorithm. The subdivision parameter `t` defines the split (default is 0.5). Either a `float` or `sympy.Rational`(recommended) can be used for `t`.

```python
# Subdivide along dimension 0 at t = 1/2.
bb_left, bb_right = bb.subdivide(0, t=sympy.Rational(1/2))
```


## License

  

This project is licensed under the [MIT License](LICENSE).

  

## Contact

  

If you have any questions or suggestions, feel free to contact me at [p.gupta@ufl.edu](mailto:p.gupta@ufl.edu).