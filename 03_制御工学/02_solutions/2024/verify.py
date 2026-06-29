#!/usr/bin/env python3
"""
AY2024 制御工学 検算
第1問: 壁-(k2∥d)-質量m-k1-入力端A(x1)。G=k1/(ms^2+ds+k1+k2)。
       ステップ応答・最大値、無減衰、(5)A固定で外力ステップ。
第2問: 図2=Gc Gp 単位帰還（一巡ボード・閉ループstep）。
       図3=1/s・Gc・1/s・Gp 三重負帰還の伝達関数とstep。
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
# 第1問  G=k1/(m s^2+d s+(k1+k2))
# ============================================================
m, d, k1, k2 = sp.symbols('m d k1 k2', positive=True)
G = k1 / (m * s**2 + d * s + (k1 + k2))
check("Q1(1) G=k1/(ms^2+ds+k1+k2)", G, k1 / (m * s**2 + d * s + k1 + k2))

# (2) m=1,d=2,k1=3,k2=2 → G=3/(s^2+2s+5), 極-1±2j, 定常3/5
G2 = G.subs({m: 1, d: 2, k1: 3, k2: 2})
check("Q1(2) G=3/(s^2+2s+5)", sp.simplify(G2 - 3 / (s**2 + 2 * s + 5)), 0)
x2 = sp.inverse_laplace_transform(G2 / s, s, t)
check("Q1(2) x2=3/5-(3/5)e^{-t}(cos2t+(1/2)sin2t)",
      sp.simplify(x2 - (sp.Rational(3, 5) - sp.Rational(3, 5) * sp.exp(-t) * (sp.cos(2 * t) + sp.Rational(1, 2) * sp.sin(2 * t)))), 0)
check("Q1(2) 定常値 3/5", sp.limit(x2, t, sp.oo), sp.Rational(3, 5))

# (3) 最大値（最初のピーク）: x2'=0 の最小正根
x2e = sp.Rational(3, 5) - sp.Rational(3, 5) * sp.exp(-t) * (sp.cos(2 * t) + sp.Rational(1, 2) * sp.sin(2 * t))
dx = sp.simplify(sp.diff(x2e, t))
# dx ∝ e^{-t} sin(2t) → ピークは 2t=π → t=π/2
check("Q1(3) dx/dt ∝ e^{-t} sin 2t", sp.simplify(dx / (sp.exp(-t) * sp.sin(2 * t))).is_constant(), True)
tmax = sp.pi / 2
x2max = sp.simplify(x2e.subs(t, tmax))
print("  [info] Q1(3) x2max =", x2max, "=", float(x2max), " at t=pi/2")
check("Q1(3) 発生時間 π/2", tmax, sp.pi / 2)
check("Q1(3) x2max=3/5+(3/5)e^{-π/2}", x2max, sp.Rational(3, 5) + sp.Rational(3, 5) * sp.exp(-sp.pi / 2))

# (4) m=1,d=0,k1=3,k2=1 → G=3/(s^2+4), x2=(3/4)(1-cos2t)
G4 = G.subs({m: 1, d: 0, k1: 3, k2: 1})
check("Q1(4) G=3/(s^2+4)", sp.simplify(G4 - 3 / (s**2 + 4)), 0)
x4 = sp.inverse_laplace_transform(G4 / s, s, t)
check("Q1(4) x2=(3/4)(1-cos2t)", sp.simplify(x4 - sp.Rational(3, 4) * (1 - sp.cos(2 * t))), 0)

# (5) A固定(x1=0), m=1,d=2,k1=1,k2=1, 外力ステップ f=1/s
#   m x2''+d x2'+(k1+k2)x2=f → x2''+2x2'+2x2=f, 定常1/2
G5 = 1 / (s**2 + 2 * s + 2)
x5 = sp.inverse_laplace_transform(G5 / s, s, t)
check("Q1(5) x2=1/2-(1/2)e^{-t}(cos t+sin t)",
      sp.simplify(x5 - (sp.Rational(1, 2) - sp.Rational(1, 2) * sp.exp(-t) * (sp.cos(t) + sp.sin(t)))), 0)
check("Q1(5) 定常値 1/2", sp.limit(x5, t, sp.oo), sp.Rational(1, 2))

# ============================================================
# 第2問
# ============================================================
# 図2: 単位帰還 forward Gc Gp, Gc=1/s, Gp=(s+1)/(10s+1)
Gc = 1 / s
Gp = (s + 1) / (10 * s + 1)
L = sp.simplify(Gc * Gp)
check("Q2(1) 一巡 L=(s+1)/(s(10s+1))", sp.simplify(L - (s + 1) / (s * (10 * s + 1))), 0)
# 低周波 |L|≈1/ω: ω=1 で 0dB の -20傾き。折れ点 0.1(極),1(零)
check("Q2(1) 低周波 |L(jω)|≈1/ω (ω=0.1→20dB)",
      20 * np.log10(abs(complex(L.subs(s, 1j * 0.001)))), 20 * np.log10(1 / 0.001), tol=0.2)

# (2) 閉ループ T=Gc Gp/(1+Gc Gp)=(s+1)/(10s^2+2s+1), 定常1（type1）
T2 = sp.simplify(Gc * Gp / (1 + Gc * Gp))
check("Q2(2) T=(s+1)/(10s^2+2s+1)", sp.simplify(T2 - (s + 1) / (10 * s**2 + 2 * s + 1)), 0)
check("Q2(2) ステップ定常値 1", sp.limit(T2, s, 0), 1)
y2 = sp.inverse_laplace_transform(T2 / s, s, t)
check("Q2(2) y(0)=0", sp.limit(y2, t, 0), 0)
check("Q2(2) y(∞)=1", sp.limit(y2, t, sp.oo), 1)
print("  [info] Q2(2) y(t) =", sp.simplify(y2))

# (3) 図3 三重負帰還の伝達関数を連立で導出
#   e1=U-nP, a=e1/s, e2=a-nQ, nP=Gc e2, e3=nP-Y, nQ=e3/s, Y=Gp nQ
U, Y, a, e2, e3, nP, nQ = sp.symbols('U Y a e2 e3 nP nQ')
K1, K2 = sp.symbols('K1 K2', positive=True)
Gc3, Gp3 = 1 / K1, 1 / K2
eqs = [
    sp.Eq(a, (U - nP) / s),
    sp.Eq(e2, a - nQ),
    sp.Eq(nP, Gc3 * e2),
    sp.Eq(e3, nP - Y),
    sp.Eq(nQ, e3 / s),
    sp.Eq(Y, Gp3 * nQ),
]
sol = sp.solve(eqs, [Y, a, e2, e3, nP, nQ], dict=True)[0]
T3 = sp.simplify(sol[Y] / U)
check("Q2(3) T=1/(K1 K2 s^2+(K1+2K2)s+1)",
      sp.simplify(T3 - 1 / (K1 * K2 * s**2 + (K1 + 2 * K2) * s + 1)), 0)

# (4) K1=1/2,K2=1/3 → T=6/((s+1)(s+6)), x=1-(6/5)e^{-t}+(1/5)e^{-6t}
T4 = T3.subs({K1: sp.Rational(1, 2), K2: sp.Rational(1, 3)})
check("Q2(4) T=6/((s+1)(s+6))", sp.simplify(T4 - 6 / ((s + 1) * (s + 6))), 0)
y4 = sp.inverse_laplace_transform(T4 / s, s, t)
check("Q2(4) y=1-(6/5)e^{-t}+(1/5)e^{-6t}",
      sp.simplify(y4 - (1 - sp.Rational(6, 5) * sp.exp(-t) + sp.Rational(1, 5) * sp.exp(-6 * t))), 0)
check("Q2(4) 定常値 1", sp.limit(y4, t, sp.oo), 1)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
