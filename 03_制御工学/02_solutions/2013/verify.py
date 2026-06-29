#!/usr/bin/env python3
"""
AY2013 制御工学 解answer検算
第1問: 一自由度振動系 m x''+d x'+k x=f。伝達関数、定常値=2 となる k、
       無減衰ステップ応答、自由応答。
第2問: K0 と G1 の直列系。ゲイン・位相、ボード(漸近線)、ステップ応答、考察。
依存: sympy, numpy, scipy.signal
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


s = sp.symbols('s')
t = sp.symbols('t', positive=True)
m, d, k = sp.symbols('m d k', positive=True)

# ============================================================
# 第1問
# ============================================================
# (1) G(s)=1/(m s^2+d s+k)
G = 1 / (m * s**2 + d * s + k)
check("Q1(1) 伝達関数 1/(m s^2+d s+k)", G, 1 / (m * s**2 + d * s + k))

# (2) m=1,d=1 単位ステップの定常値=G(0)=1/k=2 → k=1/2
G2 = G.subs({m: 1, d: 1})
yinf = sp.limit(G2 * (1 / s) * s, s, 0)  # = G2(0) = 1/k
ksol = sp.solve(sp.Eq(yinf, 2), k)
check("Q1(2) 定常値=2 となる k", ksol[0], sp.Rational(1, 2))

# (3) m=1,d=0,k=1 単位ステップ: G=1/(s^2+1) → x=1-cos t
G3 = G.subs({m: 1, d: 0, k: 1})
x3 = sp.inverse_laplace_transform(G3 / s, s, t)
check("Q1(3) ステップ応答 1-cos t", sp.simplify(x3 - (1 - sp.cos(t))), 0)
check("Q1(3) 定常的に振動（極が±j）",
      set(sp.solve(s**2 + 1, s)) == {sp.I, -sp.I}, True)

# (4) m=1,d=1,k=1, x(0)=1, x'(0)=0, f=0 自由応答
# x''+x'+x=0 → x(t)=e^{-t/2}(cos(√3/2 t)+(1/√3) sin(√3/2 t))
w = sp.sqrt(3) / 2
x4 = sp.exp(-t / 2) * (sp.cos(w * t) + (1 / sp.sqrt(3)) * sp.sin(w * t))
# 微分方程式と初期条件を満たすか
ode = sp.diff(x4, t, 2) + sp.diff(x4, t) + x4
check("Q1(4) 自由応答が ODE を満たす", sp.simplify(ode), 0)
check("Q1(4) x(0)=1", x4.subs(t, 0), 1)
check("Q1(4) x'(0)=0", sp.simplify(sp.diff(x4, t).subs(t, 0)), 0)
check("Q1(4) 定常値 0", sp.limit(x4, t, sp.oo), 0)

# ============================================================
# 第2問  G(s)=K0*G1=2/(10s^2+12.5s+2)
# ============================================================
Gs = 2 / (10 * s**2 + sp.Rational(25, 2) * s + 2)
# DC ゲイン =1 (0 dB)
check("Q2 DCゲイン G(0)=1", Gs.subs(s, 0), 1)

# (1) ゲイン・位相（ω=1 で数値照合）
w0 = 1.0
Gjw = complex(Gs.subs(s, 1j * w0))
mag = 2 / np.sqrt((2 - 10 * w0**2)**2 + (12.5 * w0)**2)
phase = -np.degrees(np.arctan2(12.5 * w0, 2 - 10 * w0**2))
check("Q2(1) |G(j1)| 公式と一致", abs(Gjw), mag, tol=1e-9)
check("Q2(1) ∠G(j1) 公式と一致", np.degrees(np.angle(Gjw)), phase, tol=1e-9)

# (2) 漸近線の折れ点 = 極の大きさ
poles = sp.solve(10 * s**2 + sp.Rational(25, 2) * s + 2, s)
pvals = sorted(abs(complex(p)) for p in poles)
check("Q2(2) 折れ点1 ω≈0.188", pvals[0], 0.18839, tol=1e-3)
check("Q2(2) 折れ点2 ω≈1.062", pvals[1], 1.06161, tol=1e-3)
check("Q2(2) 両極とも実数（過減衰）",
      all(abs(complex(p).imag) < 1e-9 for p in poles), True)

# (3) 単位ステップ応答（過減衰、定常値1）
xstep = sp.inverse_laplace_transform(Gs / s, s, t)
# 数値で初期値・定常値・代表点を確認
f = sp.lambdify(t, xstep, 'numpy')
check("Q2(3) x(0)=0", float(f(1e-9)), 0.0, tol=1e-4)
check("Q2(3) x(∞)=1", float(f(100.0)), 1.0, tol=1e-3)
# 過減衰 → 単調増加（オーバーシュートなし）: 数点で単調
ts = np.linspace(0.01, 30, 200)
xs = f(ts)
check("Q2(3) 単調増加（過減衰）", bool(np.all(np.diff(xs) > -1e-9)), True)
# ζ>1 確認
wn = sp.sqrt(sp.Rational(2, 10))
zeta = (sp.Rational(25, 2) / 10) / (2 * wn)
check("Q2(3) ζ>1（過減衰）", bool(zeta > 1), True)

note_unverifiable("Q2(4) 過渡・定常特性の考察",
                  "過減衰の遅い極が応答を支配・type0で定常偏差ありの記述問題で数値検算対象外")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
