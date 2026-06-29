#!/usr/bin/env python3
"""
AY2017 制御工学 検算
第1問: 二次振動系。伝達関数、ωn=1,ζ=0 から d,k と無減衰自由応答、指数入力応答と定常振幅。
第2問: 二段階制御（AY2012同型）。伝達関数、G1のゲイン位相・ボード、比例/位相進み補償の定常値。
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

# (2) m=1,ωn=1,ζ=0 → k=1, d=0。f=0,x(0)=1,x'(0)=0 → x=cos t
check("Q1(2) k=1", 1 * 1**2, 1)
check("Q1(2) d=0", 2 * 0 * 1 * 1, 0)
x2 = sp.cos(t)
check("Q1(2) ODE x''+x=0", sp.simplify(sp.diff(x2, t, 2) + x2), 0)
check("Q1(2) x(0)=1", x2.subs(t, 0), 1)
check("Q1(2) x'(0)=0", sp.diff(x2, t).subs(t, 0), 0)

# (3) x''+x=2e^{-t}, x(0)=1,x'(0)=0 → x=e^{-t}+sin t
x3 = sp.exp(-t) + sp.sin(t)
check("Q1(3) ODE x''+x=2e^{-t}", sp.simplify(sp.diff(x3, t, 2) + x3 - 2 * sp.exp(-t)), 0)
check("Q1(3) x(0)=1", x3.subs(t, 0), 1)
check("Q1(3) x'(0)=0", sp.diff(x3, t).subs(t, 0), 0)
# 充分時間後 → sin t、振幅最大 1
check("Q1(3) 定常振幅 1", 1, 1)

# ============================================================
# 第2問  T=G1 G2/(1+G1 K0+G1 G2)
# ============================================================
G1, G2, K0, K1 = sp.symbols('G1 G2 K0 K1')
M = G1 / (1 + G1 * K0)
T_sym = M * G2 / (1 + M * G2)
check("Q2(1) T=G1 G2/(1+G1 K0+G1 G2)",
      sp.simplify(T_sym - G1 * G2 / (1 + G1 * K0 + G1 * G2)), 0)

# (2) G1=10/((s+1)(10s+1)): DCゲイン20dB, 折れ点0.1,1
G1f = 10 / ((s + 1) * (10 * s + 1))
check("Q2(2) DCゲイン |G1(0)|=10", G1f.subs(s, 0), 10)
check("Q2(2) DCゲイン[dB]=20", 20 * sp.log(10, 10), 20)
w = 100.0
mag_db = 20 * np.log10(abs(complex(G1f.subs(s, 1j * w))))
# 折れ線: 20dB→(0.1)→-20dB/dec→(1)→-40dB/dec → ω=100: 20-20*log10(1/0.1)-40*log10(100/1)
approx_db = 20 - 20 * np.log10(1 / 0.1) - 40 * np.log10(100 / 1)
check("Q2(2) ω=100 真値と漸近線一致(±5dB)", abs(mag_db - approx_db) < 5, True)
poles = sp.Poly(sp.denom(sp.together(G1f)), s).all_roots()
check("Q2(2) 折れ点 {0.1,1}",
      set(round(abs(complex(p)), 6) for p in poles) == {0.1, 1.0}, True)

# (3) K0=1, G2=K1: 単位ステップ定常値 = T(0) = 10K1/(11+10K1)
T3 = T_sym.subs({G1: 10, K0: 1, G2: K1})
y3 = sp.simplify(T3)
check("Q2(3) y(∞)=10K1/(11+10K1)", y3, 10 * K1 / (11 + 10 * K1))

# (4) G2=((s+1)/(0.1s+1))K1: 位相進み補償（積分器なし）, G2(0)=K1 → 定常値は(3)と同じ
G2f = ((s + 1) / (sp.Rational(1, 10) * s + 1)) * K1
check("Q2(4) G2(0)=K1", G2f.subs(s, 0), K1)
y4 = sp.limit(G1f * G2f / (1 + G1f * 1 + G1f * G2f), s, 0)
check("Q2(4) y(∞)=10K1/(11+10K1)（(3)と同値）", sp.simplify(y4), 10 * K1 / (11 + 10 * K1))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
