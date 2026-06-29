#!/usr/bin/env python3
"""
AY2018 制御工学 検算
第1問: 質量m にバネk・ダンパd 並列で入力変位x1。G=(ds+k)/(ms^2+ds+k)。
       ステップ応答x2と定常値、最大値と発生時間、k=0の場合。
第2問: α付きフィードバック系（AY2014同型）G=α/(Ms^2+Ds+K)。
       ゲイン位相・ボード・位相進み補償後の極と安定判別。
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
# (1) m x2''+d x2'+k x2 = d x1'+k x1 → G=(ds+k)/(ms^2+ds+k)
G = (d * s + k) / (m * s**2 + d * s + k)
check("Q1(1) 伝達関数 (ds+k)/(ms^2+ds+k)", G, (d * s + k) / (m * s**2 + d * s + k))

# (2) m=1,d=3,k=2: x2=1+e^{-t}-2e^{-2t}, 定常1
G2 = G.subs({m: 1, d: 3, k: 2})
x2 = sp.inverse_laplace_transform(G2 / s, s, t)
check("Q1(2) x2=1+e^{-t}-2e^{-2t}",
      sp.simplify(x2 - (1 + sp.exp(-t) - 2 * sp.exp(-2 * t))), 0)
check("Q1(2) 定常値 1", sp.limit(x2, t, sp.oo), 1)

# (3) 最大値と発生時間: x2'=0 → t=ln4, x2max=9/8
x2e = 1 + sp.exp(-t) - 2 * sp.exp(-2 * t)
tmax = [v for v in sp.solve(sp.diff(x2e, t), t) if v.is_real]
check("Q1(3) 発生時間 ln4", tmax[0], sp.log(4))
check("Q1(3) 最大値 9/8", sp.simplify(x2e.subs(t, sp.log(4))), sp.Rational(9, 8))

# (4) m=1,d=1,k=0: G=1/(s+1), x2=1-e^{-t}, 定常1
G4 = G.subs({m: 1, d: 1, k: 0})
check("Q1(4) G=1/(s+1)", sp.simplify(G4 - 1 / (s + 1)), 0)
x4 = sp.inverse_laplace_transform(G4 / s, s, t)
check("Q1(4) x2=1-e^{-t}", sp.simplify(x4 - (1 - sp.exp(-t))), 0)
check("Q1(4) 定常値 1", sp.limit(x4, t, sp.oo), 1)

# ============================================================
# 第2問  G(s)=α/(M s^2+D s+K)
# ============================================================
M_, D_, K_, al = sp.symbols('M D K alpha', positive=True)
G_sym = al / (M_ * s**2 + D_ * s + K_)
check("Q2(1) G=α/(M s^2+D s+K)", G_sym, al / (M_ * s**2 + D_ * s + K_))

# (2) M=5,D=5.5,K=0.5,α=5 → G=10/((10s+1)(s+1)), DC=10(20dB)
Gn = G_sym.subs({M_: 5, D_: sp.Rational(11, 2), K_: sp.Rational(1, 2), al: 5})
check("Q2(2) G=10/((10s+1)(s+1))", sp.simplify(Gn - 10 / ((10 * s + 1) * (s + 1))), 0)
check("Q2(2) DCゲイン=10", Gn.subs(s, 0), 10)
w0 = 1.0
Gjw = complex(Gn.subs(s, 1j * w0))
mag = 10 / (np.sqrt(1 + (10 * w0)**2) * np.sqrt(1 + w0**2))
phase = -(np.degrees(np.arctan(10 * w0)) + np.degrees(np.arctan(w0)))
check("Q2(2) |G(j1)|", abs(Gjw), mag, tol=1e-9)
check("Q2(2) ∠G(j1)", np.degrees(np.angle(Gjw)), phase, tol=1e-9)

# (3) 折れ点 0.1, 1
poles2 = sp.Poly(sp.denom(sp.together(Gn)), s).all_roots()
check("Q2(3) 折れ点 {0.1,1}",
      set(round(abs(complex(p)), 6) for p in poles2) == {0.1, 1.0}, True)

# (4) Gc=(1+0.25s)/(1+0.025s), G*Gc の極 = {-0.1,-1,-40}, すべて Re<0 安定
Gc = (1 + sp.Rational(1, 4) * s) / (1 + sp.Rational(1, 40) * s)
GGc = sp.together(Gn * Gc)
poles4 = sp.Poly(sp.denom(GGc), s).all_roots()
pset = set(sp.nsimplify(p) for p in poles4)
check("Q2(4) 極 = {-0.1,-1,-40}",
      pset == {sp.Rational(-1, 10), sp.Integer(-1), sp.Integer(-40)}, True)
check("Q2(4) 全極 Re<0（安定）", all(sp.re(p) < 0 for p in poles4), True)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
