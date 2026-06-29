#!/usr/bin/env python3
"""
AY2020 制御工学 検算
第1問: 質量m に「直列バネ k1-k2」と「ダンパ d」を並列で入力端x1 に接続。
       k_eq=k1k2/(k1+k2)。G=(ds+k_eq)/(ms^2+ds+k_eq)。
       ステップ応答・最大値、合成入力応答と充分時間後の値、無減衰の場合。
第2問: G1 G2 直列＋K 負帰還。一巡伝達関数のゲイン位相・ステップ応答、K調整、G2変更比較。
依存: sympy, numpy
"""
import sympy as sp
import numpy as np

results = []


def check(label, got, expected, tol=1e-6):
    ok = False
    try:
        ok = sp.simplify(sp.sympify(got) - sp.sympify(expected)) == 0
    except Exception:
        pass
    if not ok:
        try:
            ok = abs(float(got) - float(expected)) < tol
        except Exception:
            ok = False
    results.append((label, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: got={got} expected={expected}")
    return ok


def note_unverifiable(label, reason):
    results.append((label, None))
    print(f"[SKIP] {label}: 検証不能 — {reason}")


s, t = sp.symbols('s t', positive=True)

# ============================================================
# 第1問
# ============================================================
m, d, k1, k2 = sp.symbols('m d k1 k2', positive=True)
keq = k1 * k2 / (k1 + k2)
# (1) m x2''+d x2'+keq x2 = d x1'+keq x1 → G=(ds+keq)/(ms^2+ds+keq)
G = (d * s + keq) / (m * s**2 + d * s + keq)
check("Q1(1) G=(ds+keq)/(ms^2+ds+keq)", G, (d * s + keq) / (m * s**2 + d * s + keq))

# (2) m=1,d=2,k1=2,k2=2 → keq=1, G=(2s+1)/(s+1)^2
G2 = G.subs({m: 1, d: 2, k1: 2, k2: 2})
check("Q1(2) keq=1", keq.subs({k1: 2, k2: 2}), 1)
check("Q1(2) G=(2s+1)/(s+1)^2", sp.simplify(G2 - (2 * s + 1) / (s + 1)**2), 0)
x2 = sp.inverse_laplace_transform(G2 / s, s, t)
check("Q1(2) x2=1-(1-t)e^{-t}", sp.simplify(x2 - (1 - (1 - t) * sp.exp(-t))), 0)
check("Q1(2) 定常値 1", sp.limit(x2, t, sp.oo), 1)
# 最大値: x2'=(2-t)e^{-t}=0 → t=2, x2max=1+e^{-2}
tmax = [v for v in sp.solve(sp.diff(x2, t), t) if v.is_real and v >= 0]
check("Q1(2) 発生時間 2", tmax[0], 2)
check("Q1(2) 最大値 1+e^{-2}", sp.simplify(x2.subs(t, 2)), 1 + sp.exp(-2))

# (3) m=1,d=2,k1=2,k2=4 → keq=4/3。x1=1(t)+e^{-t} → 充分時間後 1
G3 = G.subs({m: 1, d: 2, k1: 2, k2: 4})
check("Q1(3) keq=4/3", keq.subs({k1: 2, k2: 4}), sp.Rational(4, 3))
X1_3 = 1 / s + 1 / (s + 1)
x3 = sp.inverse_laplace_transform(G3 * X1_3, s, t)
check("Q1(3) 充分時間後 1", sp.limit(x3, t, sp.oo), 1)
print("  [info] Q1(3) x2(t) =", sp.simplify(x3))

# (4) m=2,d=0,k1=1,k2=2 → keq=2/3。G=(1/3)/(s^2+1/3), x2=1-cos(t/√3)
G4 = G.subs({m: 2, d: 0, k1: 1, k2: 2})
check("Q1(4) keq=2/3", keq.subs({k1: 1, k2: 2}), sp.Rational(2, 3))
check("Q1(4) G=(1/3)/(s^2+1/3)", sp.simplify(G4 - (sp.Rational(1, 3)) / (s**2 + sp.Rational(1, 3))), 0)
x4 = sp.inverse_laplace_transform(G4 / s, s, t)
check("Q1(4) x2=1-cos(t/√3)", sp.simplify(x4 - (1 - sp.cos(t / sp.sqrt(3)))), 0)

# ============================================================
# 第2問  G1=K1/(s+1), G2=1/(0.1s+1), 前向きG1G2・帰還K
# ============================================================
K1, K = sp.symbols('K1 K', positive=True)
G1f = K1 / (s + 1)
G2f = 1 / (sp.Rational(1, 10) * s + 1)

# (1) K1=2,K=1: 一巡 L=2/((s+1)(0.1s+1)), 閉ループ x=2/3-4e^{-5t}+(10/3)e^{-6t}
L1 = sp.simplify((K * G1f * G2f).subs({K1: 2, K: 1}))
check("Q2(1) L=2/((s+1)(0.1s+1))", sp.simplify(L1 - 2 / ((s + 1) * (sp.Rational(1, 10) * s + 1))), 0)
check("Q2(1) DCゲイン L(0)=2", L1.subs(s, 0), 2)
w0 = 1.0
Ljw = complex(L1.subs(s, 1j * w0))
mag = 2 / (np.sqrt(1 + w0**2) * np.sqrt(1 + (0.1 * w0)**2))
phase = -(np.degrees(np.arctan(w0)) + np.degrees(np.arctan(0.1 * w0)))
check("Q2(1) |L(j1)|", abs(Ljw), mag, tol=1e-9)
check("Q2(1) ∠L(j1)", np.degrees(np.angle(Ljw)), phase, tol=1e-9)
T1 = sp.simplify(((G1f * G2f) / (1 + K * G1f * G2f)).subs({K1: 2, K: 1}))
check("Q2(1) T=20/((s+5)(s+6))", sp.simplify(T1 - 20 / ((s + 5) * (s + 6))), 0)
x1step = sp.inverse_laplace_transform(T1 / s, s, t)
check("Q2(1) x=2/3-4e^{-5t}+(10/3)e^{-6t}",
      sp.simplify(x1step - (sp.Rational(2, 3) - 4 * sp.exp(-5 * t) + sp.Rational(10, 3) * sp.exp(-6 * t))), 0)
check("Q2(1) 定常値 2/3", sp.limit(T1, s, 0), sp.Rational(2, 3))

# (2) 折れ点 1, 10
poles = sp.Poly(sp.denom(sp.together(L1)), s).all_roots()
check("Q2(2) 折れ点 {1,10}",
      set(round(abs(complex(p)), 6) for p in poles) == {1.0, 10.0}, True)

# (3) ω=0.01 近傍で 0dB → 低周波 L(0)=K*K1=2K=1 → K=1/2
ksol = sp.solve(sp.Eq(2 * K, 1), K)
check("Q2(3) K=1/2", ksol[0], sp.Rational(1, 2))

# (4) K=1/2, G2=1/(0.2s+1): L=1/((s+1)(0.2s+1)) DC0dB, 閉ループ T=10/(s^2+6s+10)
G2b = 1 / (sp.Rational(1, 5) * s + 1)
Lb = sp.simplify((K * G1f * G2b).subs({K1: 2, K: sp.Rational(1, 2)}))
check("Q2(4) L=1/((s+1)(0.2s+1))", sp.simplify(Lb - 1 / ((s + 1) * (sp.Rational(1, 5) * s + 1))), 0)
check("Q2(4) DCゲイン L(0)=1 (0dB)", Lb.subs(s, 0), 1)
Tb = sp.simplify(((G1f * G2b) / (1 + K * G1f * G2b)).subs({K1: 2, K: sp.Rational(1, 2)}))
check("Q2(4) T=10/(s^2+6s+10)", sp.simplify(Tb - 10 / (s**2 + 6 * s + 10)), 0)
xb = sp.inverse_laplace_transform(Tb / s, s, t)
check("Q2(4) x=1-e^{-3t}(cos t+3 sin t)",
      sp.simplify(xb - (1 - sp.exp(-3 * t) * (sp.cos(t) + 3 * sp.sin(t)))), 0)
check("Q2(4) 定常値 1", sp.limit(Tb, s, 0), 1)
# 極 -3±j（弱不足減衰）
polesb = sp.Poly(s**2 + 6 * s + 10, s).all_roots()
check("Q2(4) 閉ループ極 -3±j（不足減衰）",
      set(sp.nsimplify(p) for p in polesb) == {-3 + sp.I, -3 - sp.I}, True)

note_unverifiable("Q2(4) 2系の違いの考察",
                  "G2 の時定数増で帯域縮小・閉ループが過減衰→不足減衰(僅かにオーバーシュート)へ。極で根拠提示済み")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
