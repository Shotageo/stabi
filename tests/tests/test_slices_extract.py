import numpy as np
import stabi_lem as lem

def test_compute_slices_basic():
    H, L = 25.0, 60.0
    ground = lem.make_ground_example(H, L)
    soils = [lem.Soil(18.0, 5.0, 30.0)]
    interfaces, allow = [], []
    xc, yc, R = 50.0, 40.0, 35.0
    S = lem.compute_slices_poly_multi(ground, interfaces, soils, allow, xc, yc, R, n_slices=40)
    assert S is not None
    N = len(S["x_mid"])
    for k in ["alpha", "b", "h", "W", "y_arc"]:
        assert len(S[k]) == N
        assert np.all(np.isfinite(S[k]))
    assert np.isfinite(S["dx"]) and S["dx"] > 0
    assert np.all(S["W"] > 0)

