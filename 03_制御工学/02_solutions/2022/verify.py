#!/usr/bin/env python3
"""
AY2022 制御工学 検算
第1問: 壁-バネk-質量m-ダンパd-自由端x1。G=ds/(ms^2+ds+k)。
       ステップ応答x2・最大値、減衰正弦入力 e^{-t}sin t の応答と充分時間後の値。
第2問: G1 G2 直列＋K 負帰還（AY2019同型）。一巡伝達関数・ボード・ステップ、G2&K変更比較。
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
m, d, k = sp.symbols('m d k', positive=True)
G = d * s / (m * s**2 + d * s + k)
check("Q1(1) G=ds/(ms^2+ds+k)", G, d * s / (m * s**2 + d * s + k))

# (2) m=1,d=3,k=2: x2=3(e^{-t}-e^{-2t}), 定常0
G2 = G.subs({m: 1, d: 3, k: 2})
x2 = sp.inverse_laplace_transform(G2 / s, s, t)
check("Q1(2) x2=3(e^{-t}-e^{-2t})", sp.simplify(x2 - 3 * (sp.exp(-t) - sp.exp(-2 * t))), 0)
check("Q1(2) 定常値 0", sp.limit(x2, t, sp.oo), 0)

# (3) 最大値: t=ln2, x2max=3/4
x2e = 3 * (sp.exp(-t) - sp.exp(-2 * t))
tmax = [v for v in sp.solve(sp.diff(x2e, t), t) if v.is_real and v >= 0]
check("Q1(3) 発生時間 ln2", tmax[0], sp.log(2))
check("Q1(3) 最大値 3/4", x2e.subs(t, sp.log(2)), sp.Rational(3, 4))

# (4) m=1,d=2,k=2: G=2s/((s+1)^2+1), x1=e^{-t}sin t → X1=1/((s+1)^2+1)
G4 = G.subs({m: 1, d: 2, k: 2})
check("Q1(4) G=2s/((s+1)^2+1)", sp.simplify(G4 - 2 * s / ((s + 1)**2 + 1)), 0)
X1_4 = 1 / ((s + 1)**2 + 1)
x4 = sp.inverse_laplace_transform(G4 * X1_4, s, t)
print("  [info] Q1(4) x2(t) =", sp.simplify(x4))
check("Q1(4) 充分時間後 0", sp.limit(x4, t, sp.oo), 0)
# 閉形式 x2=e^{-t}(t(sin t+cos t)-sin t)
cand = sp.exp(-t) * (t * (sp.sin(t) + sp.cos(t)) - sp.sin(t))
check("Q1(4) x2=e^{-t}(t(sin t+cos t)-sin t)",
      sp.simplify(x4 - cand), 0)

# ============================================================
# 第2問
# ============================================================
G1f = 1 / (s + 1)
G2f = 1 / (10 * s + 1)
K = 2
# (1) 一巡 L=2/((s+1)(10s+1))
L = sp.simplify(K * G1f * G2f)
check("Q2(1) L=2/((s+1)(10s+1))", sp.simplify(L - 2 / ((s + 1) * (10 * s + 1))), 0)
check("Q2(1) DCゲイン L(0)=2", L.subs(s, 0), 2)
w0 = 1.0
Ljw = complex(L.subs(s, 1j * w0))
mag = 2 / (np.sqrt(1 + w0**2) * np.sqrt(1 + (10 * w0)**2))
phase = -(np.degrees(np.arctan(w0)) + np.degrees(np.arctan(10 * w0)))
check("Q2(1) |L(j1)|", abs(Ljw), mag, tol=1e-9)
check("Q2(1) ∠L(j1)", np.degrees(np.angle(Ljw)), phase, tol=1e-9)

# (2) 折れ点 0.1, 1
poles = sp.Poly(sp.denom(sp.together(L)), s).all_roots()
check("Q2(2) 折れ点 {0.1,1}",
      set(round(abs(complex(p)), 6) for p in poles) == {0.1, 1.0}, True)

# (3) T=1/(10s^2+11s+3), 極-0.5,-0.6, x=1/3-2e^{-t/2}+(5/3)e^{-3t/5}, 定常1/3
T = sp.simplify(G1f * G2f / (1 + K * G1f * G2f))
check("Q2(3) T=1/(10s^2+11s+3)", sp.simplify(T - 1 / (10 * s**2 + 11 * s + 3)), 0)
check("Q2(3) 定常値 1/3", sp.limit(T, s, 0), sp.Rational(1, 3))
xs = sp.inverse_laplace_transform(T / s, s, t)
check("Q2(3) x=1/3-2e^{-t/2}+(5/3)e^{-3t/5}",
      sp.simplify(xs - (sp.Rational(1, 3) - 2 * sp.exp(-t / 2) + sp.Rational(5, 3) * sp.exp(-3 * t / 5))), 0)

# (4) G2=1/(0.2s+1), K=1: L=1/((s+1)(0.2s+1)) DC0dB, T=5/(s^2+6s+10), 定常1/2, 極-3±j
G2b = 1 / (sp.Rational(1, 5) * s + 1)
Kb = 1
Lb = sp.simplify(Kb * G1f * G2b)
check("Q2(4) L=1/((s+1)(0.2s+1))", sp.simplify(Lb - 1 / ((s + 1) * (sp.Rational(1, 5) * s + 1))), 0)
check("Q2(4) DCゲイン L(0)=1 (0dB)", Lb.subs(s, 0), 1)
Tb = sp.simplify(G1f * G2b / (1 + Kb * G1f * G2b))
check("Q2(4) T=5/(s^2+6s+10)", sp.simplify(Tb - 5 / (s**2 + 6 * s + 10)), 0)
check("Q2(4) 定常値 1/2", sp.limit(Tb, s, 0), sp.Rational(1, 2))
xb = sp.inverse_laplace_transform(Tb / s, s, t)
check("Q2(4) x=1/2-(1/2)e^{-3t}(cos t+3 sin t)",
      sp.simplify(xb - (sp.Rational(1, 2) - sp.Rational(1, 2) * sp.exp(-3 * t) * (sp.cos(t) + 3 * sp.sin(t)))), 0)
polesb = sp.Poly(s**2 + 6 * s + 10, s).all_roots()
check("Q2(4) 閉ループ極 -3±j（不足減衰）",
      set(sp.nsimplify(p) for p in polesb) == {-3 + sp.I, -3 - sp.I}, True)

note_unverifiable("Q2(4) 過渡特性の違いの考察",
                  "(3)は過減衰・遅い/定常1/3, (4)は不足減衰・高速&僅かにオーバーシュート/定常1/2。極で根拠提示済み")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
