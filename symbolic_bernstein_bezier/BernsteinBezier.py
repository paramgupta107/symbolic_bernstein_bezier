import sympy
import itertools
import numpy as np
import math
from typing import Union, List, Tuple


class BernsteinBezier:
    def __init__(self, coeffs: List[sympy.Basic], degrees: List[int], vars: List[sympy.Symbol]):
        """
        Initialize the Bernstein–Bézier representation.

        Parameters:
          coeffs: List of coefficients in row-major order.
          degrees: List of degrees (one per variable/dimension).
          vars: List of sympy symbols corresponding to the variables.
        """
        assert len(vars) == len(degrees), "Number of variables does not match the degrees."

        self.degrees = degrees
        shape = tuple(d + 1 for d in degrees)
        # Convert coeffs to a numpy array with object dtype and reshape it to the proper multi-dimensional shape.
        self.coeffs = np.array(coeffs, dtype=object).reshape(shape)
        self.vars = vars

    def _bernstein_polynomial(self, i: int, degree: int, t: sympy.Symbol) -> sympy.Basic:
        """
        Calculate the Bernstein polynomial of degree 'degree' at
        parameter 't' for a given index 'i'.
        """
        return sympy.binomial(degree, i) * (t ** i) * ((1 - t) ** (degree - i))

    def to_expr(self):
        """
        Evaluate a function in BB form with arbitrary number of variables and degrees.

        Returns:
            sympy.Basic: The evaluated expression.
        """
        result = sympy.Integer(0)
        # Iterate over multi-indices using numpy.ndindex.
        for idx in np.ndindex(*self.coeffs.shape):
            monomial = sympy.Integer(1)
            for var_idx, i_val in enumerate(idx):
                monomial *= self._bernstein_polynomial(i_val, self.degrees[var_idx], self.vars[var_idx])
            result += monomial * self.coeffs[idx]
        return result

    @classmethod
    def _mono_to_bb(cls, u: sympy.Symbol, degree: int, bb_degree: int) -> sympy.Basic:
        """
        Converts a monomial u^k to an expression using sympy symbols representing Bernstein Polynomials.
        The sympy symbol for B(i,n)(u) is named B_i_n_u.
        """
        result = 0
        denom = sympy.binomial(bb_degree, degree)
        for i in range(degree, bb_degree + 1):
            coeff = sympy.binomial(i, degree) / denom
            bp = sympy.Symbol(f"B_{i}_{bb_degree}_{u}")
            result += coeff * bp
        return result

    @classmethod
    def _polynomial_to_bb_form(cls, poly: sympy.Basic,
                               vars: List[sympy.Symbol]) -> Tuple[sympy.Basic, List[int]]:
        """
        Convert a polynomial expression into an expression of symbolic Bernstein-basis placeholders for each variable.

        Args:
            poly (sympy.Basic): A Sympy expression in many variables.
            vars (List[sympy.Symbol]): The subset of Sympy symbols we want to convert into Bernstein basis.

        Returns:
            Tuple[sympy.Basic, List[int]]: A tuple containing the expression involving placeholders 
            (named B_i_deg_varName for each variable) and the list of maximum degrees.
        """
        poly_expanded = sympy.expand(poly)
        P = sympy.poly(poly_expanded, vars)
        max_degs = [P.degree(var) for var in vars]
        result = sympy.Integer(0)
        for monomial_powers, coeff in P.terms():
            factor_expr = sympy.Integer(1)
            for var, power, deg in zip(vars, monomial_powers, max_degs):
                factor_expr *= cls._mono_to_bb(var, power, deg)
            result += coeff * factor_expr
        return result, max_degs

    @classmethod
    def _extract_control_points_bb(cls, poly: sympy.Basic, vars: List[sympy.Symbol]) -> Tuple[np.ndarray, List[int]]:
        """
        Given an expression in Bernstein Bézier form, extract the control points as a multi-dimensional
        numpy array in row-major ordering. Each Bernstein polynomial must be represented by a sympy symbol
        named B_i_deg_varName.

        Args:
            poly (sympy.Basic): An expression in Bernstein Bézier form.
            vars (List[sympy.Symbol]): The subset of sympy symbols to convert into Bernstein basis.

        Returns:
            tuple: (numpy array of control points, list of degrees)
        """
        expr, max_degs = cls._polynomial_to_bb_form(poly, vars)
        # Create a list of symbolic Bernstein placeholders for each variable.
        bp = [[sympy.Symbol(f"B_{i}_{deg}_{var}") for i in range(deg + 1)]
              for var, deg in zip(vars, max_degs)]
        bp_flat = list(itertools.chain.from_iterable(bp))
        poly_obj = sympy.Poly(expr, bp_flat, expand=True)
        coeffs = []
        for combo in itertools.product(*(range(d + 1) for d in max_degs)):
            monomial = sympy.Integer(1)
            for var_idx, i_val in enumerate(combo):
                monomial *= bp[var_idx][i_val]
            coeffs.append(poly_obj.coeff_monomial(monomial))
        # Convert the list of coefficients to a numpy array and reshape it.
        coeffs_array = np.array(coeffs, dtype=object).reshape(tuple(d + 1 for d in max_degs))
        return coeffs_array, max_degs

    @classmethod
    def from_expr(cls, expr: sympy.Basic, vars: List[sympy.Symbol]) -> "BernsteinBezier":
        """
        Convert a sympy expression to a Bernstein–Bézier representation.

        Parameters:
          expr: The sympy expression to convert.
          vars: List of sympy symbols.

        Returns:
          A BernsteinBezier instance if conversion is successful.
        """
        coeffs, degrees = cls._extract_control_points_bb(expr, vars)
        return cls(coeffs, degrees, vars)
    
    def get_coeff(self, index: Tuple[int]) -> sympy.Basic:
        """
        Return the coefficient at the given multi-index.

        Args:
            index (tuple): A tuple of indices.

        Returns:
            sympy.Basic: The coefficient at that index.
        """
        assert len(index) == len(self.degrees), "Number of indices does not match the number of variables."
        return self.coeffs[index]

    def get_all_coeffs(self) -> list:
        """
        Return all coefficients as a flat list in row-major order.
        """
        return list(self.coeffs.flatten())
    
    def _convolve(self, other: "BernsteinBezier") -> "BernsteinBezier":
        """
        Convolve two Bernstein Bézier representations.

        Args:
            other (BernsteinBezier): The other Bernstein Bézier representation.

        Returns:
            BernsteinBezier: The result of the convolution.
        """
        assert self.vars == other.vars, "Variables do not match."
        assert len(self.degrees) == len(other.degrees), "Number of dimensions do not match."

        # Determine output shape: each dimension's size is (size1 + size2 - 1)
        out_shape = tuple(a + b - 1 for a, b in zip(self.coeffs.shape, other.coeffs.shape))
        new_coeffs = np.zeros(out_shape, dtype=object)
        
        # Direct multi-dimensional convolution: iterate over all indices of self.coeffs and other.coeffs.
        for idx in np.ndindex(*self.coeffs.shape):
            for jdx in np.ndindex(*other.coeffs.shape):
                out_idx = tuple(i + j for i, j in zip(idx, jdx))
                new_coeffs[out_idx] += self.coeffs[idx] * other.coeffs[jdx]
        
        # The new degree in each dimension is (new_dim_size - 1)
        new_degrees = [dim - 1 for dim in out_shape]
        return BernsteinBezier(new_coeffs.flatten(), new_degrees, self.vars)

    
    def _binom_array(self):
        """
        Given a shape (a tuple of dimensions), returns an np.array of that shape 
        where each element is the product over dimensions of binom(shape_dim-1, idx).
        
        For example, for shape (2, 3), the returned array has entries:
        
            (0 1).(0 2)  (0 1).(1 2)  (0 1).(2 2)
            (1 1).(0 2)  (1 1).(1 2)  (1 1).(2 2)
        
        Here, (i n) represents binom(n, i).
        """
        shape = self.coeffs.shape
        # For each dimension, create a 1D array of binomial coefficients:
        binom_lists = [np.array([math.comb(dim - 1, i) for i in range(dim)])
                    for dim in shape]
        
        # Use np.ix_ to create broadcastable arrays and then multiply them together:
        grids = np.ix_(*binom_lists)
        
        # Multiply over all dimensions:
        result = np.ones(shape, dtype=int)
        for g in grids:
            result *= g  # broadcasting multiplies element-wise
        
        return result
    
    def _scale_for_mul(self, scale_up: bool) -> "BernsteinBezier":
        """
        Scale the Bernstein Bézier coefficients for multiplication.

        Args:
            scale_up (bool): Whether to scale up or down.

        Returns:
            BernsteinBezier: The scaled Bernstein Bézier representation.
        """
        scale = self._binom_array()
        if scale_up:
            new_coeffs = self.coeffs * scale
        else:
            new_coeffs = self.coeffs / scale
        return BernsteinBezier(new_coeffs, self.degrees, self.vars)
    
    def __mul__(self, other: "BernsteinBezier") -> "BernsteinBezier":
        """
        Multiply two Bernstein Bézier representations.

        Args:
            other (BernsteinBezier): The other Bernstein Bézier representation.

        Returns:
            BernsteinBezier: The result of the multiplication.
        """
        assert type(other) == BernsteinBezier, "Can only multiply with another BernsteinBezier."
        assert self.vars == other.vars, "Variables do not match."
        assert len(self.degrees) == len(other.degrees), "Number of dimensions do not match."

        # Scale the coefficients up for multiplication.
        self_scaled = self._scale_for_mul(True)
        other_scaled = other._scale_for_mul(True)

        # Perform convolution and scale down the result.
        result = self_scaled._convolve(other_scaled)
        # result = other_scaled._convolve(self_scaled)
        return result._scale_for_mul(False)
    
    def degree_raise(self, raise_degrees: List[int]) -> "BernsteinBezier":
        """
        Raise the degree of the Bernstein Bézier representation in each dimension.

        Args:
            raise_degrees (List[int]): The number of degrees to raise in each dimension.

        Returns:
            BernsteinBezier: The result of the degree raise.
        """
        assert len(raise_degrees) == len(self.degrees), "Number of dimensions do not match."
        assert all(raise_deg >= 0 for raise_deg in raise_degrees), "Degree raises must be non-negative."
        if all(raise_deg == 0 for raise_deg in raise_degrees):
            return self
        mul_coeffs = np.ones(tuple(d + 1 for d in raise_degrees))
        mul_bb = BernsteinBezier(mul_coeffs, raise_degrees, self.vars)
        return self * mul_bb

    def __add__(self, other: "BernsteinBezier") -> "BernsteinBezier":
        """
        Add two Bernstein Bézier representations. If the degrees do not match, the result will have the maximum degree at each dimension.
        
        Args:
            other (BernsteinBezier): The other Bernstein Bézier representation.
            
        Returns:
            BernsteinBezier: The result of the addition.
        """
        assert type(other) == BernsteinBezier, "Can only add with another BernsteinBezier."
        assert self.vars == other.vars, "Variables do not match."
        max_degrees = [max(a, b) for a, b in zip(self.degrees, other.degrees)]
        self_raise = self.degree_raise([max_deg - deg for deg, max_deg in zip(self.degrees, max_degrees)])
        other_raise = other.degree_raise([max_deg - deg for deg, max_deg in zip(other.degrees, max_degrees)])
        new_coeffs = self_raise.coeffs + other_raise.coeffs
        return BernsteinBezier(new_coeffs, max_degrees, self.vars)

    def __sub__(self, other: "BernsteinBezier") -> "BernsteinBezier":
        """
        Subtrace two Bernstein Bézier representations. If the degrees do not match, the result will have the maximum degree at each dimension.
        
        Args:
            other (BernsteinBezier): The other Bernstein Bézier representation.
            
        Returns:
            BernsteinBezier: The result of the addition.
        """
        assert type(other) == BernsteinBezier, "Can only subtract with another BernsteinBezier."
        assert self.vars == other.vars, "Variables do not match."
        max_degrees = [max(a, b) for a, b in zip(self.degrees, other.degrees)]
        self_raise = self.degree_raise([max_deg - deg for deg, max_deg in zip(self.degrees, max_degrees)])
        other_raise = other.degree_raise([max_deg - deg for deg, max_deg in zip(other.degrees, max_degrees)])
        new_coeffs = self_raise.coeffs - other_raise.coeffs
        return BernsteinBezier(new_coeffs, max_degrees, self.vars)

    def __eq__(self, other: "BernsteinBezier") -> bool:
        """
        Check if two Bernstein Bézier representations are equal.

        Args:
            other (BernsteinBezier): The other Bernstein Bézier representation.

        Returns:
            bool: True if the two Bernstein Bézier representations are equal.
        """
        assert type(other) == BernsteinBezier, "Can only compare with another BernsteinBezier."
        assert self.vars == other.vars, "Variables do not match."
        if self.degrees == other.degrees:
            return np.array_equal(self.coeffs, other.coeffs)
        return sympy.simplify(self.to_expr() - other.to_expr()) == 0
    
    def differentiate(self, dim: int) -> "BernsteinBezier":
        """
        Differentiate the Bernstein Bézier representation with respect to a variable.

        Args:
            dim (int): The index of the variable to differentiate.

        Returns:
            BernsteinBezier: The result of the differentiation.
        """
        assert 0 <= dim < len(self.vars), "Invalid variable index."
        if self.degrees[dim] == 0:
            return BernsteinBezier(np.zeros_like(self.coeffs), self.degrees, self.vars)
        new_coeffs = np.diff(self.coeffs, axis=dim) * self.degrees[dim]
        new_degrees = self.degrees.copy()
        new_degrees[dim] -= 1
        return BernsteinBezier(new_coeffs, new_degrees, self.vars)
    
    def extract_boundary(self, dim: int, index: int) -> "BernsteinBezier":
        """
        Extract the boundary of the Bernstein Bézier representation at the given index.

        Args:
            dim (int): The index of the variable to extract the boundary from.
            index (int): The index of the boundary.

        Returns:
            BernsteinBezier: The result of the boundary extraction.
        """
        assert 0 <= dim < len(self.vars), "Invalid variable index."
        # Slice the coefficients along the given dimension.
        new_coeffs = np.take(self.coeffs, index, axis=dim)
        # new_coeffs = np.squeeze(new_coeffs, axis=dim)
        new_degrees = self.degrees[:dim] + self.degrees[dim + 1:]
        return BernsteinBezier(new_coeffs, new_degrees, self.vars[:dim] + self.vars[dim + 1:])

    def compose(self, funcs: List[Union["BernsteinBezier", sympy.Basic]], vars: List[sympy.Symbol]) -> "BernsteinBezier":
        """
        Compose the Bernstein Bézier representation with another Bernstein Bézier or a sympy expression.

        Args:
            funcs (List[Union[BernsteinBezier, sympy.Basic]]): The Bernstein Bézier or sympy expression to compose with.

        Returns:
            BernsteinBezier: The result of the composition.
        """
        assert len(funcs) == len(self.vars), "Number of composed functions does not match the number of variables."
        composed_vars = [func.to_expr() if isinstance(func, BernsteinBezier) else func for func in funcs]
        conposed_expr = self.to_expr().subs(list(zip(self.vars, composed_vars)))
        return BernsteinBezier.from_expr(conposed_expr, vars)

    def integrate(self, dim: int) -> Union["BernsteinBezier", sympy.Basic]:
        """
        Integrate the Bernstein Bézier representation with respect to a variable by summing the coefficents and dividing by the number of coefficients.

        Args:
            dim (int): The index of the variable to integrate.

        Returns:
             Union["BernsteinBezier", sympy.Basic]: The result of the integration. If there is only one dimension left, return a sympy expression.
        """
        assert 0 <= dim < len(self.vars), "Invalid variable index."
        new_coeffs = np.sum(self.coeffs, axis=dim) / (self.degrees[dim] + 1)
        new_degrees = self.degrees[:dim] + self.degrees[dim + 1:]
        if len(new_degrees) == 0:
            return new_coeffs
        new_vars = self.vars[:dim] + self.vars[dim + 1:]
        return BernsteinBezier(new_coeffs, new_degrees, new_vars)

    def subdivide(self, dim: int, t:Union[float, sympy.Rational] = sympy.Rational(1, 2)) -> Tuple["BernsteinBezier", "BernsteinBezier"]:
        """
        Subdivide the Bernstein Bézier representation along the given dimension using De Casteljau's algorithm.
        
        Args:
            dim (int): The index of the variable (dimension) along which to subdivide.
            t Union[float, sympy.Rational]: The parameter at which to subdivide (default is 0.5).
        
        Returns:
            tuple[BernsteinBezier, BernsteinBezier]: A tuple (left, right) where 'left' is the BernsteinBezier
            representing the sub-curve for t in [0, t] and 'right' for t in [t, 1] along the specified dimension.
        """
        assert 0 <= dim < len(self.vars), "Invalid variable index."
        shape = self.coeffs.shape
        n = shape[dim]  # number of control points along the given dimension (degree+1)
        
        # Prepare arrays for the left and right subdivided control points.
        left_coeffs = self.coeffs.copy()
        right_coeffs = self.coeffs.copy()

        currLayer = self.coeffs.copy()
        # We will build new layers along the subdivided dimension.
        for i in range(1, n):
            index_left = tuple(slice(0, -1) if d == dim else slice(None) for d in range(len(shape)))
            index_right = tuple(slice(1, None) if d == dim else slice(None) for d in range(len(shape)))
            left = currLayer[index_left]
            right = currLayer[index_right]
            currLayer = (1 - t) * left + t * right
            
            # Build index tuples to assign the leftmost and rightmost points from currLayer.
            leftLayerIdx = tuple(0 if d == dim else slice(None) for d in range(len(shape)))
            rightLayerIdx = tuple(-1 if d == dim else slice(None) for d in range(len(shape)))
            # The index in the original array where we store these boundary values:
            leftIdx = list(slice(None) for _ in range(len(shape)))
            rightIdx = list(slice(None) for _ in range(len(shape)))
            leftIdx[dim] = i
            rightIdx[dim] = n - 1 - i
            leftIdx = tuple(leftIdx)
            rightIdx = tuple(rightIdx)
            
            left_coeffs[leftIdx] = currLayer[leftLayerIdx]
            right_coeffs[rightIdx] = currLayer[rightLayerIdx]
        
        left_bb = BernsteinBezier(left_coeffs.flatten(), self.degrees, self.vars)
        right_bb = BernsteinBezier(right_coeffs.flatten(), self.degrees, self.vars)
        return left_bb, right_bb

    def simplify_coeffs(self) -> "BernsteinBezier":
        """
        Simplify the coefficients of the Bernstein Bézier representation.

        Returns:
            BernsteinBezier: The result of the simplification.
        """
        new_coeffs = np.vectorize(lambda x: sympy.simplify(x))(self.coeffs)
        new_degrees = self.degrees.copy()
        return BernsteinBezier(new_coeffs, new_degrees, self.vars)

    def to_matrix(self, vars: List[sympy.Symbol]) -> np.ndarray:
        """
        Generate a matrix to represent the Bernstein Bézier coefficients with respect to given vars.
        The size of the matrix will be (number of vars * number of coefficients).
        vars * matrix will yield the coefficients in row-major order.
        """
        num_vars = len(vars)
        num_coeffs = np.prod([d + 1 for d in self.degrees])
        matrix = np.zeros((num_vars, num_coeffs), dtype=object)
        # Fill the matrix with coefficients.
        for idx in np.ndindex(*self.coeffs.shape):
            coeff = self.coeffs[idx]
            row = np.array([coeff.coeff(vars[i]) for i in range(num_vars)])
            ## add the row to the matrix at the appropriate index.
            matrix[:, np.ravel_multi_index(idx, self.coeffs.shape)] = row
        return matrix
    def __repr__(self):
        return f"BernsteinBezier(coeffs={self.coeffs}, degrees={self.degrees})"
