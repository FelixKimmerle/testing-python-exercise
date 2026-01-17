"""
Tests for functionality checks in class SolveDiffusion2D
"""

import pytest
import numpy as np
from diffusion2d import SolveDiffusion2D

@pytest.fixture
def solver():
    solver = SolveDiffusion2D()
    return solver

def test_initialize_physical_parameters(solver):
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    w = 20.
    h = 15.
    dx = 0.2
    dy = 0.3
    d = 5.
    T_cold = 200.
    T_hot = 500.

    solver.initialize_domain(w,h,dx,dy)
    solver.initialize_physical_parameters(d,T_cold,T_hot)

    expected_dt = 0.00276923076923077

    assert solver.dt == pytest.approx(expected_dt), "dt should be approximately {expected_dt}"


def test_set_initial_condition(solver):
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    w = 20.
    h = 15.
    dx = 0.2
    dy = 0.3
    d = 5.
    T_cold = 200.
    T_hot = 500.

    solver.initialize_domain(w,h,dx,dy)
    solver.initialize_physical_parameters(d,T_cold,T_hot)
    computed_u = solver.set_initial_condition()

    expected_u = solver.T_cold * np.ones((solver.nx, solver.ny))

    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = solver.T_hot

    assert np.array_equal(expected_u, computed_u), "Initial condition array is not equal to the expected"

