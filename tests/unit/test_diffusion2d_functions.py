"""
Tests for functions in class SolveDiffusion2D
"""

import numpy as np
import pytest
from diffusion2d import SolveDiffusion2D


def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    w = 20.
    h = 15.
    dx = 0.2
    dy = 0.3
    expected_nx  = 100
    expected_ny = 50
    solver = SolveDiffusion2D()
    solver.initialize_domain(w,h,dx,dy)
    assert solver.nx == pytest.approx(expected_nx), "nx should be 100"
    assert solver.ny == pytest.approx(expected_ny), "ny should be 50"



def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    solver.dx = 0.2
    solver.dy = 0.3
    d = 5.
    T_cold = 200.
    T_hot =  500.

    expected_dt = 0.00276923076923077
    solver.initialize_physical_parameters(d, T_cold, T_hot)
    assert solver.dt == pytest.approx(expected_dt), "dt should be approximately {expected_dt}"
    assert solver.T_cold == pytest.approx(T_cold), "T_cold should be 200"
    assert solver.T_hot == pytest.approx(T_hot), "T_hot should be 500"

def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    solver.dx = 0.2
    solver.dy = 0.3
    solver.nx = 100
    solver.ny = 50
    solver.T_cold = 200.
    solver.T_hot = 500.
    solver.dt = 0.00276923076923077
    solver.D = 5.
    
    expected_u = solver.T_cold * np.ones((solver.nx, solver.ny))

    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                expected_u[i, j] = solver.T_hot

    computed_u = solver.set_initial_condition()
    assert np.array_equal(expected_u, computed_u), "Initial condition array is not equal to the expected"
