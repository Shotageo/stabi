import stabi_lem as lem

def test_min_fs_baseline():
    H, L = 25.0, 60.0
    ground = lem.make_ground_example(H, L)
    soils = [lem.Soil(18.0, 5.0, 30.0)]
    interfaces = []
    allow = []
    # テスト用の代表円弧（固定）
    xc, yc, R = 50.0, 40.0, 35.0
    Fs = lem.fs_given_R_multi(ground, interfaces, soils, allow,
                              "Bishop (simplified)", xc, yc, R, n_slices=40)
    assert Fs is not None
    # 初回は Fs をログに出して一度だけ置き換え
    expected = 1.234567  # ← CIログの値に差し替え
    assert abs(Fs - expected) < 1e-6

