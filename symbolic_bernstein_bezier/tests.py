import unittest
import sympy
from BernsteinBezier import BernsteinBezier
import itertools
import numpy as np

class TestBernsteinBezierConversion(unittest.TestCase):
    def test_1d(self):
        # One dimension: degree = [2] (3 control points)
        degree = [2]
        t = sympy.symbols("t")
        # Create coefficient symbols: c0, c1, c2
        coeffs = sympy.symbols("c0:3")
        bb = BernsteinBezier(coeffs, degree, vars=[t])
        expr = bb.to_expr()
        new_expr = 2 * expr
        new_bb = BernsteinBezier.from_expr(new_expr, vars=[t])
        for idx in itertools.product(range(degree[0] + 1)):
            original = bb.get_coeff(idx)
            new_val = new_bb.get_coeff(idx)
            self.assertTrue(sympy.simplify(new_val - 2 * original) == 0,
                            f"1D: At index {idx}, expected {2 * original}, got {new_val}")

    def test_2d(self):
        # Two dimensions: degree = [1, 1] (4 control points)
        degree = [1, 1]
        t1, t2 = sympy.symbols("t1 t2")
        # Create coefficient symbols: c00, c01, c10, c11 (row-major order)
        coeffs = []
        for i in range(2):
            for j in range(2):
                coeffs.append(sympy.Symbol(f"c{i}{j}"))
        bb = BernsteinBezier(coeffs, degree, vars=[t1, t2])
        expr = bb.to_expr()
        new_expr = 2 * expr
        new_bb = BernsteinBezier.from_expr(new_expr, vars=[t1, t2])
        for idx in itertools.product(*(range(d + 1) for d in degree)):
            original = bb.get_coeff(idx)
            new_val = new_bb.get_coeff(idx)
            self.assertTrue(sympy.simplify(new_val - 2 * original) == 0,
                            f"2D: At index {idx}, expected {2 * original}, got {new_val}")

    def test_4d(self):
        # Four dimensions: degree = [1, 1, 1, 1] (16 control points)
        degree = [1, 1, 1, 1]
        t1, t2, t3, t4 = sympy.symbols("t1 t2 t3 t4")
        coeffs = []
        # Generate names like c0000, c0001, ... c1111.
        for idx in itertools.product(*(range(d + 1) for d in degree)):
            name = "c" + "".join(str(i) for i in idx)
            coeffs.append(sympy.Symbol(name))
        bb = BernsteinBezier(coeffs, degree, vars=[t1, t2, t3, t4])
        expr = bb.to_expr()
        new_expr = 2 * expr
        new_bb = BernsteinBezier.from_expr(new_expr, vars=[t1, t2, t3, t4])
        for idx in itertools.product(*(range(d + 1) for d in degree)):
            original = bb.get_coeff(idx)
            new_val = new_bb.get_coeff(idx)
            self.assertTrue(sympy.simplify(new_val - 2 * original) == 0,
                            f"4D: At index {idx}, expected {2 * original}, got {new_val}")
            
class TestBernsteinBezierConvolution(unittest.TestCase):
    def test_convolution_degrees(self):
        # Define degrees for two 4D BernsteinBezier objects (different degrees).
        degrees1 = [1, 2, 3, 4]  # This BernsteinBezier has shape (2, 3, 4, 5)
        degrees2 = [2, 1, 4, 3]  # This one has shape (3, 2, 5, 4)
        
        # Create symbolic coefficients for bb1.
        coeffs1 = [sympy.Symbol("a" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees1))]
        # Create symbolic coefficients for bb2.
        coeffs2 = [sympy.Symbol("b" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees2))]
        
        # Define 4 symbolic variables.
        t1, t2, t3, t4 = sympy.symbols("t1 t2 t3 t4")
        vars = [t1, t2, t3, t4]
        
        # Create two BernsteinBezier objects.
        bb1 = BernsteinBezier(coeffs1, degrees1, vars)
        bb2 = BernsteinBezier(coeffs2, degrees2, vars)
        
        # Perform convolution.
        bb_conv = bb1._convolve(bb2)
        # print(bb_conv)
        
        # The expected degrees should be the element-wise sum of the original degrees.
        expected_degrees = [degrees1[i] + degrees2[i] for i in range(4)]
        
        self.assertEqual(bb_conv.degrees, expected_degrees,
                         f"Expected degrees {expected_degrees}, got {bb_conv.degrees}")

class TestBernsteinBezierMultiplication(unittest.TestCase):
    def test_multiplication_expr_equivalence(self):
        # Define degrees for two BBs with 3 variables (different degrees)
        degrees1 = [1, 2, 1]  # BB1: shape (2, 3, 2)
        degrees2 = [2, 1, 2]  # BB2: shape (3, 2, 3)
        t1, t2, t3 = sympy.symbols("t1 t2 t3")
        vars = [t1, t2, t3]
        
        # Generate symbolic coefficients for BB1
        coeffs1 = [
            sympy.Symbol("a" + "".join(str(i) for i in idx))
            for idx in itertools.product(*(range(d + 1) for d in degrees1))
        ]
        # Generate symbolic coefficients for BB2
        coeffs2 = [
            sympy.Symbol("b" + "".join(str(i) for i in idx))
            for idx in itertools.product(*(range(d + 1) for d in degrees2))
        ]
        
        # Create the BernsteinBezier instances
        bb1 = BernsteinBezier(coeffs1, degrees1, vars)
        bb2 = BernsteinBezier(coeffs2, degrees2, vars)
        
        # Multiply using the implemented __mul__ method and convert to expression
        product_bb = bb1 * bb2
        product_expr = product_bb.to_expr()
        
        # Multiply the expressions of the original BBs directly
        expr1 = bb1.to_expr()
        expr2 = bb2.to_expr()
        direct_product_expr = expr1 * expr2

        diff = sympy.simplify(product_expr - direct_product_expr)
        
        
        # Check if the two expressions are equivalent
        self.assertTrue(
            diff == 0,
            f"Expected product expressions to be equal"
        )

class TestDegreeRaise(unittest.TestCase):
    def test_degree_raise_4d(self):
        # Original degrees for 4 dimensions and the degree raise amounts.
        orig_degrees = [1, 2, 1, 2]
        raise_degrees = [2, 1, 3, 0]
        expected_degrees = [o + r for o, r in zip(orig_degrees, raise_degrees)]
        
        # Create 4 symbolic variables.
        vars = sympy.symbols("t1 t2 t3 t4")
        
        # Generate symbolic coefficients for the original BB.
        coeffs = [sympy.Symbol("c" + "".join(str(i) for i in idx))
                  for idx in itertools.product(*(range(d + 1) for d in orig_degrees))]
        
        # Create the original BernsteinBezier instance.
        bb = BernsteinBezier(coeffs, orig_degrees, vars)
        
        # Perform degree raising.
        bb_raised = bb.degree_raise(raise_degrees)
        
        # Convert both the original and raised BB to sympy expressions.
        expr_original = sympy.simplify(bb.to_expr())
        expr_raised = sympy.simplify(bb_raised.to_expr())
        
        # Check that the polynomial expressions are equivalent.
        self.assertTrue(sympy.simplify(expr_raised - expr_original) == 0,
                        f"Expected expressions to be equivalent.")
        
        # Check that the raised degrees are correct.
        self.assertEqual(bb_raised.degrees, expected_degrees,
                         f"Expected degrees {expected_degrees}, got {bb_raised.degrees}")
        
class TestBernsteinBezierAddSub(unittest.TestCase):
    def test_add_same_degrees(self):
        # Use 2 dimensions with same degrees.
        degrees = [2, 2]
        t1, t2 = sympy.symbols("t1 t2")
        vars = [t1, t2]
        # Create symbolic coefficients for bb1 and bb2.
        coeffs1 = [sympy.Symbol("a" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees))]
        coeffs2 = [sympy.Symbol("b" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees))]
        bb1 = BernsteinBezier(coeffs1, degrees, vars)
        bb2 = BernsteinBezier(coeffs2, degrees, vars)
        
        result = bb1 + bb2
        expected_expr = sympy.simplify(bb1.to_expr() + bb2.to_expr())
        result_expr = sympy.simplify(result.to_expr())
        
        self.assertTrue(sympy.simplify(result_expr - expected_expr) == 0,
                        "Addition with same degrees failed.")
        self.assertEqual(result.degrees, degrees,
                         f"Expected degrees {degrees}, got {result.degrees}")

    def test_add_different_degrees(self):
        # Use 2 dimensions with different degrees.
        degrees1 = [1, 2]
        degrees2 = [2, 1]
        expected_degrees = [max(d1, d2) for d1, d2 in zip(degrees1, degrees2)]  # [2, 2]
        t1, t2 = sympy.symbols("t1 t2")
        vars = [t1, t2]
        coeffs1 = [sympy.Symbol("a" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees1))]
        coeffs2 = [sympy.Symbol("b" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees2))]
        bb1 = BernsteinBezier(coeffs1, degrees1, vars)
        bb2 = BernsteinBezier(coeffs2, degrees2, vars)
        
        result = bb1 + bb2
        expected_expr = sympy.simplify(bb1.to_expr() + bb2.to_expr())
        result_expr = sympy.simplify(result.to_expr())
        
        self.assertTrue(sympy.simplify(result_expr - expected_expr) == 0,
                        "Addition with different degrees failed.")
        self.assertEqual(result.degrees, expected_degrees,
                         f"Expected degrees {expected_degrees}, got {result.degrees}")

    def test_sub_same_degrees(self):
        # Use 2 dimensions with same degrees.
        degrees = [3, 3]
        t1, t2 = sympy.symbols("t1 t2")
        vars = [t1, t2]
        coeffs1 = [sympy.Symbol("a" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees))]
        coeffs2 = [sympy.Symbol("b" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees))]
        bb1 = BernsteinBezier(coeffs1, degrees, vars)
        bb2 = BernsteinBezier(coeffs2, degrees, vars)
        
        result = bb1 - bb2
        expected_expr = sympy.simplify(bb1.to_expr() - bb2.to_expr())
        result_expr = sympy.simplify(result.to_expr())
        
        self.assertTrue(sympy.simplify(result_expr - expected_expr) == 0,
                        "Subtraction with same degrees failed.")
        self.assertEqual(result.degrees, degrees,
                         f"Expected degrees {degrees}, got {result.degrees}")

    def test_sub_different_degrees(self):
        # Use 2 dimensions with different degrees.
        degrees1 = [1, 3]
        degrees2 = [2, 2]
        expected_degrees = [max(d1, d2) for d1, d2 in zip(degrees1, degrees2)]  # [2, 3]
        t1, t2 = sympy.symbols("t1 t2")
        vars = [t1, t2]
        coeffs1 = [sympy.Symbol("a" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees1))]
        coeffs2 = [sympy.Symbol("b" + "".join(str(i) for i in idx))
                   for idx in itertools.product(*(range(d + 1) for d in degrees2))]
        bb1 = BernsteinBezier(coeffs1, degrees1, vars)
        bb2 = BernsteinBezier(coeffs2, degrees2, vars)
        
        result = bb1 - bb2
        expected_expr = sympy.simplify(bb1.to_expr() - bb2.to_expr())
        result_expr = sympy.simplify(result.to_expr())
        
        self.assertTrue(sympy.simplify(result_expr - expected_expr) == 0,
                        "Subtraction with different degrees failed.")
        self.assertEqual(result.degrees, expected_degrees,
                         f"Expected degrees {expected_degrees}, got {result.degrees}")
        
class TestBernsteinBezierEquality(unittest.TestCase):
    def test_eq_same_degrees(self):
        # Create a 4-dimensional BernsteinBezier with fixed degrees.
        degrees = [1, 2, 1, 2]
        t1, t2, t3, t4 = sympy.symbols("t1 t2 t3 t4")
        vars = [t1, t2, t3, t4]
        # Generate coefficient symbols in row-major order.
        coeffs = [sympy.Symbol("c" + "".join(str(i) for i in idx))
                  for idx in itertools.product(*(range(d + 1) for d in degrees))]
        bb1 = BernsteinBezier(coeffs, degrees, vars)
        bb2 = BernsteinBezier(coeffs, degrees, vars)
        
        # They should be equal.
        self.assertTrue(bb1 == bb2, "BBs with the same degrees and coefficients should be equal.")
        
        # Modify one coefficient in a copy and check that they are no longer equal.
        coeffs_modified = list(coeffs)
        coeffs_modified[0] = coeffs_modified[0] + 1  # change first coefficient
        bb_modified = BernsteinBezier(coeffs_modified, degrees, vars)
        self.assertFalse(bb1 == bb_modified, "BBs with different coefficients should not be equal.")

    def test_eq_different_degrees(self):
        # Create a 4-dimensional BernsteinBezier with given degrees.
        orig_degrees = [1, 2, 1, 2]
        t1, t2, t3, t4 = sympy.symbols("t1 t2 t3 t4")
        vars = [t1, t2, t3, t4]
        coeffs = [sympy.Symbol("c" + "".join(str(i) for i in idx))
                  for idx in itertools.product(*(range(d + 1) for d in orig_degrees))]
        bb1 = BernsteinBezier(coeffs, orig_degrees, vars)
        
        # Degree-raise bb1 to new degrees.
        raise_degrees = [1, 1, 1, 1]  # Amount to raise in each dimension.
        bb2 = bb1.degree_raise(raise_degrees)
        
        # Although the degrees differ, they represent the same function.
        self.assertTrue(bb1 == bb2, "Degree raised BB should be equal to the original function representation.")
        
        # Modify one coefficient in the degree raised BB and assert inequality.
        coeffs_raised = bb2.get_all_coeffs()
        coeffs_raised[0] = coeffs_raised[0] + 1  # Change one coefficient.
        bb3 = BernsteinBezier(coeffs_raised, bb2.degrees, vars)
        self.assertFalse(bb1 == bb3, "Modified degree raised BB should not be equal to the original.")

class TestBernsteinBezierDifferentiate(unittest.TestCase):
    def test_differentiate_nonzero(self):
        # Create a 2D BernsteinBezier with nonzero degree in both dimensions.
        degrees = [2, 2]
        vars = sympy.symbols("t1 t2")
        coeffs = [sympy.Symbol("c" + "".join(str(i) for i in idx))
                  for idx in itertools.product(*(range(d + 1) for d in degrees))]
        bb = BernsteinBezier(coeffs, degrees, vars)
        
        # Differentiate with respect to dimension 0 (variable t1).
        bb_diff = bb.differentiate(0)
        expr_original = sympy.simplify(bb.to_expr())
        expr_diff = sympy.simplify(bb_diff.to_expr())
        
        # Compute the expected derivative via sympy differentiation.
        expected_diff = sympy.diff(expr_original, vars[0])
        
        self.assertTrue(sympy.simplify(expr_diff - expected_diff) == 0,
                        f"Expected derivative {expected_diff}, got {expr_diff}")

    def test_differentiate_zero_degree(self):
        # Create a 2D BernsteinBezier where the second dimension has degree 0.
        degrees = [2, 0]
        t1, t2 = sympy.symbols("t1 t2")
        vars = [t1, t2]
        coeffs = [sympy.Symbol("c" + "".join(str(i) for i in idx))
                  for idx in itertools.product(*(range(d + 1) for d in degrees))]
        bb = BernsteinBezier(coeffs, degrees, vars)
        
        # Differentiate with respect to dimension 1 (variable t2), which has degree 0.
        bb_diff = bb.differentiate(1)
        expr_diff = sympy.simplify(bb_diff.to_expr())
        
        # Compute the expected derivative via sympy differentiation.
        expected_diff = sympy.diff(bb.to_expr(), vars[1])

        self.assertTrue(sympy.simplify(expr_diff - expected_diff) == 0,
                        f"Expected derivative {expected_diff}, got {expr_diff}")

class TestExtractBoundary(unittest.TestCase):
    def setUp(self):
        # Set up a 2D BernsteinBezier for previous tests.
        self.degrees = [1, 2]
        self.t1, self.t2 = sympy.symbols("t1 t2")
        self.vars_2d = [self.t1, self.t2]
        coeffs_2d = [sympy.Symbol("c" + "".join(str(i) for i in idx))
                     for idx in itertools.product(*(range(d + 1) for d in self.degrees))]
        self.bb_2d = BernsteinBezier(coeffs_2d, self.degrees, self.vars_2d)

    def test_extract_boundary_positive_index(self):
        # Extract boundary along dimension 1 (columns) at index 0 (first column).
        bb_boundary = self.bb_2d.extract_boundary(1, 0)
        # Expected coefficients are the first column: [c00, c10]
        expected_coeffs = [sympy.Symbol("c00"), sympy.Symbol("c10")]
        self.assertEqual(bb_boundary.get_all_coeffs(), expected_coeffs,
                         f"Expected coefficients {expected_coeffs}, got {bb_boundary.get_all_coeffs()}")
        self.assertEqual(bb_boundary.degrees, [self.degrees[0]],
                         f"Expected degrees {[self.degrees[0]]}, got {bb_boundary.degrees}")
        self.assertEqual(bb_boundary.vars, [self.t1],
                         f"Expected vars {[self.t1]}, got {bb_boundary.vars}")

    def test_extract_boundary_negative_index(self):
        # Extract boundary along dimension 1 at index -1 (last column).
        bb_boundary = self.bb_2d.extract_boundary(1, -1)
        # Expected coefficients are the last column: [c02, c12]
        expected_coeffs = [sympy.Symbol("c02"), sympy.Symbol("c12")]
        self.assertEqual(bb_boundary.get_all_coeffs(), expected_coeffs,
                         f"Expected coefficients {expected_coeffs}, got {bb_boundary.get_all_coeffs()}")
        self.assertEqual(bb_boundary.degrees, [self.degrees[0]],
                         f"Expected degrees {[self.degrees[0]]}, got {bb_boundary.degrees}")
        self.assertEqual(bb_boundary.vars, [self.t1],
                         f"Expected vars {[self.t1]}, got {bb_boundary.vars}")

    def test_extract_boundary_3d(self):
        # Create a 3D BernsteinBezier with degrees [1, 1, 2] (shape: 2×2×3).
        degrees_3d = [1, 1, 2]
        t1, t2, t3 = sympy.symbols("t1 t2 t3")
        vars_3d = [t1, t2, t3]
        coeffs_3d = [sympy.Symbol("c" + "".join(str(i) for i in idx))
                     for idx in itertools.product(*(range(d + 1) for d in degrees_3d))]
        bb_3d = BernsteinBezier(coeffs_3d, degrees_3d, vars_3d)
        
        # Extract boundary along dimension 2 (the third dimension) at index -1 (last index).
        bb_boundary = bb_3d.extract_boundary(2, -1)
        
        # Expected: coefficients where the third index is fixed to the last value (2).
        # The remaining coefficients (for dimensions 0 and 1) are arranged in row-major order:
        # For i in range(2) and j in range(2), coefficient "c" + str(i) + str(j) + "2".
        expected_coeffs = [sympy.Symbol(f"c{i}{j}2") for i in range(2) for j in range(2)]
        
        self.assertEqual(bb_boundary.get_all_coeffs(), expected_coeffs,
                         f"Expected coefficients {expected_coeffs}, got {bb_boundary.get_all_coeffs()}")
        # The new degrees should be the original degrees with the extracted dimension removed.
        self.assertEqual(bb_boundary.degrees, [1, 1],
                         f"Expected degrees [1, 1], got {bb_boundary.degrees}")
        # The new vars should be the remaining variables after removing the one at index 2.
        self.assertEqual(bb_boundary.vars, [t1, t2],
                         f"Expected vars {[t1, t2]}, got {bb_boundary.vars}")

class TestBernsteinBezierCompose(unittest.TestCase):
    def test_compose_with_sympy_subs(self):
        # Create a BernsteinBezier representing f(t1, t2) = t1 + t2.
        t1, t2, u = sympy.symbols("t1 t2 u")
        # Build f from the expression.
        f_expr = t1 + t2
        bb = BernsteinBezier.from_expr(f_expr, [t1, t2])
        
        composed_bb = bb.compose([u**2, u**3], [u])
        
        expected_expr = u**2 + u**3
        # Check that the composed BernsteinBezier (converted to expression) equals expected_expr.
        self.assertTrue(sympy.simplify(composed_bb.to_expr() - expected_expr) == 0,
                        f"Expected composed expression {expected_expr}, got {composed_bb.to_expr()}")

    def test_compose_with_mixed_subs(self):
        # Create a BernsteinBezier representing f(t1, t2) = t1 - t2.
        t1, t2, u = sympy.symbols("t1 t2 u")
        f_expr = t1 - t2
        bb = BernsteinBezier.from_expr(f_expr, [t1, t2])
        
        # For substitution: use a BernsteinBezier for t1 and a sympy expression for t2.
        # Let g(t) = 2*u, represented as a BernsteinBezier in variable u.
        g_expr = 2 * u
        bb_sub = BernsteinBezier.from_expr(g_expr, [u])
        # For t2, substitute with u**2.
        substitution_t2 = u**2
        
        # Compose: f(t1, t2) with t1 -> g(u) and t2 -> u**2.
        composed_bb = bb.compose([bb_sub, substitution_t2], [u])
        
        # Expected: f(2*u, u**2) = 2*u - u**2.
        expected_expr = 2 * u - u**2
        self.assertTrue(sympy.simplify(composed_bb.to_expr() - expected_expr) == 0,
                        f"Expected composed expression {expected_expr}, got {composed_bb.to_expr()}")

class TestBernsteinBezierIntegrate(unittest.TestCase):
    def test_integrate_1d(self):
        # Create a 1D BernsteinBezier with degree 2 (3 control points)
        degree = [2]
        t = sympy.symbols("t")
        vars = [t]
        coeffs = [sympy.Symbol(f"a{i}") for i in range(3)]
        bb = BernsteinBezier(coeffs, degree, vars)
        
        # Convert the BernsteinBezier to an expression.
        expr = sympy.simplify(bb.to_expr())
        # Compute the definite integral over [0,1] with respect to t.
        integral_expr = sympy.integrate(expr, (t, 0, 1))
        
        # Use the integrate method to get the integrated result.
        integrated = bb.integrate(0)
        # For 1D, integrate returns a sympy expression.
        self.assertTrue(sympy.simplify(integral_expr - integrated) == 0,
                        "Integral not computed correctly")

    def test_integrate_2d_dim(self):
        # Create a 2D BernsteinBezier with degrees [1, 2] (shape: 2×3)
        degrees = [1, 2]
        t1, t2 = sympy.symbols("t1 t2")
        vars = [t1, t2]
        coeffs = [sympy.Symbol(f"a{idx[0]}{idx[1]}")
                  for idx in itertools.product(range(degrees[0] + 1), range(degrees[1] + 1))]
        bb = BernsteinBezier(coeffs, degrees, vars)
        
        # Original expression.
        expr = sympy.simplify(bb.to_expr())
        # Compute the definite integral with respect to t1 (dimension 0).
        integral_expr = sympy.integrate(expr, (t2, 0, 1))
        
        # Integrate along dimension 0.
        bb_int = bb.integrate(1)
        # bb_int is a 1D BernsteinBezier; convert it to expression.
        result_expr = sympy.simplify(bb_int.to_expr())
        
        self.assertTrue(sympy.simplify(integral_expr - result_expr) == 0,
                        "Integral not computed correctly")

class TestBernsteinBezierSubdivide(unittest.TestCase):
    def test_subdivide_1d(self):
        # f(t) = t**2, with t as the parameter.
        t = sympy.symbols("t")
        f_expr = t**4 + t
        bb = BernsteinBezier.from_expr(f_expr, [t])
        t0 = sympy.Rational(1, 2)  # use rational parameter to avoid FP issues

        left, right = bb.subdivide(0, t=t0)
        
        # For the left branch, the reparameterized function is f(t0*t)
        expected_left = sympy.simplify(f_expr.subs(t, t0 * t))
        left_expr = sympy.simplify(left.to_expr())
        self.assertTrue(sympy.simplify(left_expr - expected_left) == 0,
                        f"1D left subdivision: expected {expected_left}, got {left_expr}")
        
        # For the right branch, the reparameterized function is f(t0 + (1-t0)*t)
        expected_right = sympy.simplify(f_expr.subs(t, t0 + (1 - t0) * t))
        right_expr = sympy.simplify(right.to_expr())
        self.assertTrue(sympy.simplify(right_expr - expected_right) == 0,
                        f"1D right subdivision: expected {expected_right}, got {right_expr}")

    def test_subdivide_2d(self):
        # f(t1, t2) = t1*t2, with parameters t1 and t2.
        t1, t2 = sympy.symbols("t1 t2")
        f_expr = t1**2 * t2**4
        bb = BernsteinBezier.from_expr(f_expr, [t1, t2])
        t0 = sympy.Rational(1, 3)  # choose a rational subdivision parameter
        
        # Subdivide along dimension 0.
        left, right = bb.subdivide(0, t=t0)
        
        # Left: f(t1, t2) becomes f(t0*t1, t2)
        expected_left = sympy.simplify(f_expr.subs(t1, t0 * t1))
        left_expr = sympy.simplify(left.to_expr())
        self.assertTrue(sympy.simplify(left_expr - expected_left) == 0,
                        f"2D left subdivision (dim0): expected {expected_left}, got {left_expr}")
        
        # Right: f(t1, t2) becomes f(t0 + (1-t0)*t1, t2)
        expected_right = sympy.simplify(f_expr.subs(t1, t0 + (1 - t0) * t1))
        right_expr = sympy.simplify(right.to_expr())
        self.assertTrue(sympy.simplify(right_expr - expected_right) == 0,
                        f"2D right subdivision (dim0): expected {expected_right}, got {right_expr}")

    def test_subdivide_3d_dim1(self):
        # f(t1, t2, t3) = t1 + t2 + t3, with parameters t1, t2, t3.
        t1, t2, t3 = sympy.symbols("t1 t2 t3")
        f_expr = t1**3 + t2**2 + t3**2
        bb = BernsteinBezier.from_expr(f_expr, [t1, t2, t3])
        t0 = sympy.Rational(1, 2)
        
        # Subdivide along dimension 1.
        left, right = bb.subdivide(1, t=t0)
        
        # Left: reparameterize t2 -> t0*t2.
        expected_left = sympy.simplify(f_expr.subs(t2, t0 * t2))
        left_expr = sympy.simplify(left.to_expr())
        self.assertTrue(sympy.simplify(left_expr - expected_left) == 0,
                        f"3D left subdivision (dim1): expected {expected_left}, got {left_expr}")
        
        # Right: reparameterize t2 -> t0 + (1-t0)*t2.
        expected_right = sympy.simplify(f_expr.subs(t2, t0 + (1 - t0) * t2))
        right_expr = sympy.simplify(right.to_expr())
        self.assertTrue(sympy.simplify(right_expr - expected_right) == 0,
                        f"3D right subdivision (dim1): expected {expected_right}, got {right_expr}")

class TestSimplifyCoeffs(unittest.TestCase):
    def test_simplify_coeffs(self):
        x = sympy.symbols("x", real=True)
        # Define unsimplified coefficients (e.g., (x**2 - 1)/(x - 1) should simplify to x + 1)
        coeffs = [ (x**2 - 1)/(x - 1), (x**2 - 1)/(x - 1) + 2, (x**2 - 1)/(x - 1) - 3 ]
        degrees = [2]
        vars = [x]
        bb = BernsteinBezier(coeffs, degrees, vars)
        bb_simpl = bb.simplify_coeffs()
        self.assertTrue(all(sympy.simplify(c - s) == 0 for c, s in zip(coeffs, bb_simpl.get_all_coeffs())),
                        "Coefficients not simplified correctly")

class TestToMatrix(unittest.TestCase):
    def test_to_matrix(self):
        u, v = sympy.symbols("u v")
        a, b, c, d = sympy.symbols("a b c d")
        coeffs = [a+b, b+c, c+d, d+a,
                b+c, c+d, d+a, a+b,
                c+d, d+a, a+b, b+c,
                d+a, a+b, b+c, c+d]

        degrees = [3, 3]
        vars = [u, v]
        bb = BernsteinBezier(coeffs, degrees, vars)

        matrix = bb.to_matrix((a, b, c, d))
        expected = np.array([[1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
                            [1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
                            [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1]])
        self.assertTrue(np.array_equal(matrix, expected),
                            f"Expected matrix:\n{expected}\nGot:\n{matrix}")

if __name__ == "__main__":
    unittest.main()
