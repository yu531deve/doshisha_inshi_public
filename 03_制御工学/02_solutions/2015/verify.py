#!/usr/bin/env python3
"""
AY2015 制御工学 検算
第1問: 二次振動系。伝達関数、ωn/ζ から d,k と定常値、無減衰自由応答、指数入力応答と定常振幅。
第2問: G2=K0 G1。ゲイン位相・ボード・ステップ応答、G3 帰還系で K1 決定と過渡特性比較。
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
# (1) G=1/(m s^2+d s+k)
check("Q1(1) 伝達関数", 1 / (m * s**2 + d * s + k), 1 / (m * s**2 + d * s + k))

# (2) m=1, ωn=1, ζ=0.5 → k=ωn^2 m=1, d=2ζωn m=1。単位ステップ定常値=1/k=1
wn, zeta = 1, sp.Rational(1, 2)
kval = wn**2 * 1
dval = 2 * zeta * wn * 1
check("Q1(2) k", kval, 1)
check("Q1(2) d", dval, 1)
xinf = sp.limit(1 / (s**2 + dval * s + kval), s, 0)  # G(0)=1/k
check("Q1(2) 定常値 1（初期値に無関係）", xinf, 1)

# (3) m=1,d=0,k=1, f=0, x(0)=1,x'(0)=0 → x=cos t
x3 = sp.cos(t)
check("Q1(3) ODE x''+x=0", sp.simplify(sp.diff(x3, t, 2) + x3), 0)
check("Q1(3) x(0)=1", x3.subs(t, 0), 1)
check("Q1(3) x'(0)=0", sp.diff(x3, t).subs(t, 0), 0)

# (4) x''+x=e^{-t}, x(0)=1,x'(0)=0 → x=1/2 e^{-t}+1/2 cos t+1/2 sin t
x4 = sp.Rational(1, 2) * sp.exp(-t) + sp.Rational(1, 2) * sp.cos(t) + sp.Rational(1, 2) * sp.sin(t)
check("Q1(4) ODE x''+x=e^{-t}", sp.simplify(sp.diff(x4, t, 2) + x4 - sp.exp(-t)), 0)
check("Q1(4) x(0)=1", x4.subs(t, 0), 1)
check("Q1(4) x'(0)=0", sp.diff(x4, t).subs(t, 0), 0)
# 充分時間後の定常項 (1/2)(cos t+sin t) の振幅 = √2/2
amp = sp.sqrt((sp.Rational(1, 2))**2 + (sp.Rational(1, 2))**2)
check("Q1(4) 定常振幅 √2/2", amp, sp.sqrt(2) / 2)

# ============================================================
# 第2問
# ============================================================
# (1) G2=K0 G1=2/(20s^2+22s+2)=1/((10s+1)(s+1))
G2 = 2 / (20 * s**2 + 22 * s + 2)
check("Q2(1) G2=1/((10s+1)(s+1))", sp.simplify(G2 - 1 / ((10 * s + 1) * (s + 1))), 0)
check("Q2(1) DCゲイン=1", G2.subs(s, 0), 1)
w0 = 1.0
Gjw = complex(G2.subs(s, 1j * w0))
mag = 1 / (np.sqrt(1 + (10 * w0)**2) * np.sqrt(1 + w0**2))
phase = -(np.degrees(np.arctan(10 * w0)) + np.degrees(np.arctan(w0)))
check("Q2(1) |G2(j1)|", abs(Gjw), mag, tol=1e-9)
check("Q2(1) ∠G2(j1)", np.degrees(np.angle(Gjw)), phase, tol=1e-9)

# (2) 折れ点 0.1, 1
poles = sp.Poly(sp.denom(sp.together(G2)), s).all_roots()
check("Q2(2) 折れ点 {0.1,1}",
      set(round(abs(complex(p)), 6) for p in poles) == {0.1, 1.0}, True)

# (3) 単位ステップ応答 x=1 -10/9 e^{-0.1t}+1/9 e^{-t}（過減衰、定常1）
x_step = sp.inverse_laplace_transform(G2 / s, s, t)
f = sp.lambdify(t, x_step, 'numpy')
check("Q2(3) x(0)=0", float(f(1e-9)), 0.0, tol=1e-4)
check("Q2(3) x(∞)=1", float(f(200.0)), 1.0, tol=1e-3)
ts = np.linspace(0.01, 80, 300)
check("Q2(3) 単調増加（過減衰）", bool(np.all(np.diff(f(ts)) > -1e-9)), True)
# 解析係数の確認: residues
check("Q2(3) 係数 -10/9 (極-0.1)",
      sp.limit((s + sp.Rational(1, 10)) * G2 / s, s, -sp.Rational(1, 10)),
      -sp.Rational(10, 9))
check("Q2(3) 係数 1/9 (極-1)",
      sp.limit((s + 1) * G2 / s, s, -1), sp.Rational(1, 9))

# (4) フィードバック T=K1 G2/(1+G2 G3), G3=s+1
K1 = sp.symbols('K1', positive=True)
G3 = s + 1
T = sp.simplify(K1 * G2 / (1 + G2 * G3))
check("Q2(4) T=K1/(2(s+1)(5s+1))",
      sp.simplify(T - K1 / (2 * (s + 1) * (5 * s + 1))), 0)
# 定常値 T(0)=K1/2 を (3)の定常値1 と一致 → K1=2
k1sol = sp.solve(sp.Eq(T.subs(s, 0), 1), K1)
check("Q2(4) K1=2", k1sol[0], 2)
# K1=2 のとき極 = -1, -0.2（G2の遅い極 -0.1 → -0.2 に高速化）
Tn = T.subs(K1, 2)
polesT = sp.Poly(sp.denom(sp.together(Tn)), s).all_roots()
check("Q2(4) 閉ループ極 {-1,-0.2}",
      set(sp.nsimplify(p) for p in polesT) == {sp.Integer(-1), sp.Rational(-1, 5)}, True)
check("Q2(4) 支配時定数 10s→5s（高速化）",
      1 / sp.Rational(1, 5), 5)  # 遅い極 -0.2 の時定数 5s（G2 は -0.1 で 10s）

note_unverifiable("Q2(4) 過渡特性の比較考察",
                  "支配時定数が半減し整定が速くなる旨の記述で、数値根拠は上記極で提示済み")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
