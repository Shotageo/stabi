import stabi_lem as lem

# 初回は expected=None にしてログへ BASELINE_FS を出す
EXPECTED = None  # ここに数値を入れたら厳密比較に切り替わる

def test_min_fs_baseline():
    H, L = 25.0, 60.0
    ground = lem.make_ground_example(H, L)
    soils = [lem.Soil(18.0, 5.0, 30.0)]
    interfaces, allow = [], []
    xc, yc, R = 50.0, 40.0, 35.0

    Fs = lem.fs_given_R_multi(
        ground, interfaces, soils, allow,
        "Bishop (simplified)", xc, yc, R, n_slices=40
    )
    assert Fs is not None
    print(f"BASELINE_FS={Fs:.9f}")

    if EXPECTED is None:
        # まだ基準値未確定：ここでは失敗させずログだけ残す
        assert True
    else:
        assert abs(Fs - EXPECTED) < 1e-6
