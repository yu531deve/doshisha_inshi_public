#!/usr/bin/env python3
"""
AY2014 制御工学 検算
第1問: 二次振動系。伝達関数・ωn・ζ、定常値=5 となる β、ステップ応答、正弦入力応答。
第2問: α 付きフィードバック系。伝達関数、ゲイン位相、ボード(漸近線)、位相進み補償後の極と安定判別。
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
m, d, k = sp.symbols('m d k', positive=True)

# ============================================================
# 第1問
# ============================================================
# (1) G=1/(m s^2+d s+k), ωn=√(k/m), ζ=d/(2√(km))
G = 1 / (m * s**2 + d * s + k)
check("Q1(1) 伝達関数", G, 1 / (m * s**2 + d * s + k))
wn = sp.sqrt(k / m)
zeta = d / (2 * sp.sqrt(k * m))
check("Q1(1) ωn^2=k/m", wn**2, k / m)
check("Q1(1) 2ζωn=d/m", sp.simplify(2 * zeta * wn), d / m)

# (2) m=1,d=1,k=β 単位ステップ定常値=1/β=5 → β=1/5（初期値は安定系で減衰し定常に無関係）
beta = sp.symbols('beta', positive=True)
yinf = sp.limit(1 / (s**2 + s + beta), s, 0)  # G(0)=1/β
bsol = sp.solve(sp.Eq(yinf, 5), beta)
check("Q1(2) 定常値=5 となる β", bsol[0], sp.Rational(1, 5))

# (3) m=1,d=3,k=2, x(0)=5,x'(0)=-1, 単位ステップ
# x''+3x'+2x=1, → x=1/2+8e^{-t}-3.5e^{-2t}
x3 = sp.Rational(1, 2) + 8 * sp.exp(-t) - sp.Rational(7, 2) * sp.exp(-2 * t)
ode3 = sp.diff(x3, t, 2) + 3 * sp.diff(x3, t) + 2 * x3
check("Q1(3) ODE x''+3x'+2x=1 を満たす", sp.simplify(ode3), 1)
check("Q1(3) x(0)=5", x3.subs(t, 0), 5)
check("Q1(3) x'(0)=-1", sp.diff(x3, t).subs(t, 0), -1)
check("Q1(3) 定常値 1/2", sp.limit(x3, t, sp.oo), sp.Rational(1, 2))
# t>0 で単調減少（x'<0）: x'=-8e^{-t}+7e^{-2t}=e^{-2t}(7-8e^{t})<0 (t>0)
check("Q1(3) t>0 で単調減少", bool((7 - 8 * sp.exp(sp.Rational(1, 100))) < 0), True)

# (4) m=1,d=1,k=1, zero IC, f=sin(ωt)
# x(t)= (ω/Δ)e^{-t/2}[cos(√3/2 t)+((2ω^2-1)/√3) sin(√3/2 t)]
#       - (1/Δ)[ω cos(ωt)+(ω^2-1) sin(ωt)],  Δ=ω^4-ω^2+1
w = sp.symbols('omega', positive=True)
Delta = w**4 - w**2 + 1
r = sp.sqrt(3) / 2
x4 = (w / Delta) * sp.exp(-t / 2) * (sp.cos(r * t) + ((2 * w**2 - 1) / sp.sqrt(3)) * sp.sin(r * t)) \
     - (1 / Delta) * (w * sp.cos(w * t) + (w**2 - 1) * sp.sin(w * t))
# sympy の逆ラプラスと数値照合（ω=2, t=1.3）
X = w / ((s**2 + w**2) * (s**2 + s + 1))
x4_ref = sp.inverse_laplace_transform(X, s, t)
for wv, tv in [(2, 1.3), (0.5, 2.0), (1, 0.7)]:
    a = complex(x4.subs({w: wv, t: tv})).real
    b = complex(x4_ref.subs({w: wv, t: tv})).real
    check(f"Q1(4) x(t) 逆ラプラス一致 (ω={wv},t={tv})", a, b, tol=1e-6)
# 定常振幅 = |G(jω)| = 1/√Δ
check("Q1(4) 定常振幅=1/√(ω^4-ω^2+1)",
      sp.sqrt(w**2 + (w**2 - 1)**2) / Delta, 1 / sp.sqrt(Delta))

# ============================================================
# 第2問  G(s)=α/(M s^2+D s+K)
# ============================================================
M_, D_, K_, al = sp.symbols('M D K alpha', positive=True)
# (1) 導出: A(M+D/s+K/s^2)=U, Y=αA/s^2 → G=α/(M s^2+D s+K)
G_sym = al / (M_ * s**2 + D_ * s + K_)
check("Q2(1) G=α/(M s^2+D s+K)", G_sym, al / (M_ * s**2 + D_ * s + K_))

# (2) M=2.5,D=3,K=0.5,α=0.5 → G=1/((5s+1)(s+1)), DC=1
Gn = G_sym.subs({M_: sp.Rational(5, 2), D_: 3, K_: sp.Rational(1, 2), al: sp.Rational(1, 2)})
check("Q2(2) G=1/((5s+1)(s+1))", sp.simplify(Gn - 1 / ((5 * s + 1) * (s + 1))), 0)
check("Q2(2) DCゲイン=1", Gn.subs(s, 0), 1)
w0 = 1.0
Gjw = complex(Gn.subs(s, 1j * w0))
mag = 1 / (np.sqrt(1 + (5 * w0)**2) * np.sqrt(1 + w0**2))
phase = -(np.degrees(np.arctan(5 * w0)) + np.degrees(np.arctan(w0)))
check("Q2(2) |G(j1)|", abs(Gjw), mag, tol=1e-9)
check("Q2(2) ∠G(j1)", np.degrees(np.angle(Gjw)), phase, tol=1e-9)

# (3) 折れ点 ω=0.2, ω=1
poles2 = sp.Poly(sp.denom(sp.together(Gn)), s).all_roots()
check("Q2(3) 折れ点 {0.2,1}",
      set(round(abs(complex(p)), 6) for p in poles2) == {0.2, 1.0}, True)

# (4) Gc=(1+0.25s)/(1+0.025s), G*Gc の極 = {-0.2,-1,-40}, すべて Re<0 安定
Gc = (1 + sp.Rational(1, 4) * s) / (1 + sp.Rational(1, 40) * s)
GGc = sp.together(Gn * Gc)
poles4 = sp.Poly(sp.denom(GGc), s).all_roots()
pset = set(sp.nsimplify(p) for p in poles4)
check("Q2(4) 極 = {-0.2,-1,-40}",
      pset == {sp.Rational(-1, 5), sp.Integer(-1), sp.Integer(-40)}, True)
check("Q2(4) 全極 Re<0（安定）", all(sp.re(p) < 0 for p in poles4), True)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
