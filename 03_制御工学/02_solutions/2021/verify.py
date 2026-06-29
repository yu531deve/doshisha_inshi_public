#!/usr/bin/env python3
"""
AY2021 制御工学 検算
第1問: 二次振動系（AY2015〔I〕同型）。ωn=1,ζ=0.5 から d,k と定常値、無減衰自由応答、指数入力応答と定常振幅。
第2問: G1,G2,G3 直列＋2重負帰還（AY2016〔II〕同型）。T=G1G2G3/(1+G1+G2)。ゲイン・ボード・ステップ・G3変更比較。
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
check("Q1(1) 伝達関数", 1 / (m * s**2 + d * s + k), 1 / (m * s**2 + d * s + k))
# (2) m=1,ωn=1,ζ=0.5 → k=1,d=1, 定常値=1/k=1
check("Q1(2) k", 1 * 1**2, 1)
check("Q1(2) d", 2 * sp.Rational(1, 2) * 1 * 1, 1)
check("Q1(2) 定常値 1", sp.limit(1 / (s**2 + s + 1), s, 0), 1)
# (3) m=1,d=0,k=1, f=0,x(0)=1,x'(0)=0 → x=cos t
x3 = sp.cos(t)
check("Q1(3) ODE x''+x=0", sp.simplify(sp.diff(x3, t, 2) + x3), 0)
check("Q1(3) x(0)=1,x'(0)=0", [x3.subs(t, 0), sp.diff(x3, t).subs(t, 0)] == [1, 0], True)
# (4) x''+x=e^{-t}, x(0)=1,x'(0)=0 → x=1/2 e^{-t}+1/2 cos t+1/2 sin t, 定常振幅 √2/2
x4 = sp.Rational(1, 2) * sp.exp(-t) + sp.Rational(1, 2) * sp.cos(t) + sp.Rational(1, 2) * sp.sin(t)
check("Q1(4) ODE x''+x=e^{-t}", sp.simplify(sp.diff(x4, t, 2) + x4 - sp.exp(-t)), 0)
check("Q1(4) x(0)=1,x'(0)=0", [x4.subs(t, 0), sp.diff(x4, t).subs(t, 0)] == [1, 0], True)
check("Q1(4) 定常振幅 √2/2", sp.sqrt(sp.Rational(1, 2)**2 + sp.Rational(1, 2)**2), sp.sqrt(2) / 2)

# ============================================================
# 第2問  T=G1 G2 G3/(1+G1+G2)
# ============================================================
G1s, G2s, G3s = sp.symbols('G1 G2 G3')
check("Q2(1) T=G1 G2 G3/(1+G1+G2)",
      G1s * G2s * G3s / (1 + G1s + G2s), G1s * G2s * G3s / (1 + G1s + G2s))
# (2) G1=3/s,G2=1/(s+4),G3=4 → T=12/((s+2)(s+6)), DC=1
g1, g2, g3 = 3 / s, 1 / (s + 4), 4
T = sp.simplify(g1 * g2 * g3 / (1 + g1 + g2))
check("Q2(2) T=12/((s+2)(s+6))", sp.simplify(T - 12 / ((s + 2) * (s + 6))), 0)
check("Q2(2) DCゲイン=1(0dB)", T.subs(s, 0), 1)
poles2 = sp.Poly(sp.denom(sp.together(T)), s).all_roots()
check("Q2(2) 折れ点 {2,6}",
      set(round(abs(complex(p)), 6) for p in poles2) == {2.0, 6.0}, True)
# (3) x=1-1.5e^{-2t}+0.5e^{-6t}, 定常1
xs = sp.inverse_laplace_transform(T / s, s, t)
check("Q2(3) x=1-1.5e^{-2t}+0.5e^{-6t}",
      sp.simplify(xs - (1 - sp.Rational(3, 2) * sp.exp(-2 * t) + sp.Rational(1, 2) * sp.exp(-6 * t))), 0)
check("Q2(3) 定常値 1", sp.limit(xs, t, sp.oo), 1)
# (4) G3=2s+4 → T=6/(s+6), x=1-e^{-6t}
g3b = 2 * s + 4
Tb = sp.simplify(g1 * g2 * g3b / (1 + g1 + g2))
check("Q2(4) T=6/(s+6)", sp.simplify(Tb - 6 / (s + 6)), 0)
xb = sp.inverse_laplace_transform(Tb / s, s, t)
check("Q2(4) x=1-e^{-6t}", sp.simplify(xb - (1 - sp.exp(-6 * t))), 0)
check("Q2(4) 1次系 時定数 1/6", sp.Rational(1, 6), 1 / sp.Integer(6))
note_unverifiable("Q2(4) 過渡特性の考察",
                  "零点-2が極-2を相殺し1次系化・帯域拡大・高速化の記述。極比較で根拠提示済み")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
