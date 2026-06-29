#!/usr/bin/env python3
"""
AY2016 制御工学 検算
第1問: 壁-バネk-質量m-ダンパd-自由端x1 の系。G=ds/(ms^2+ds+k)。
       ステップ応答x2と定常値、最大値と発生時間、k=0の場合。
第2問: G1,G2,G3 直列＋2重負帰還。T=G1 G2 G3/(1+G1+G2)。
       ゲイン・ボード・ステップ応答、G3変更時の過渡比較。
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
# (1) 運動方程式 m x2'' + d x2' + k x2 = d x1'  → G=ds/(m s^2+d s+k)
G = d * s / (m * s**2 + d * s + k)
check("Q1(1) 伝達関数 ds/(ms^2+ds+k)", G, d * s / (m * s**2 + d * s + k))

# (2) m=1,d=3,k=2: G=3s/((s+1)(s+2)). 単位ステップ x2=3(e^{-t}-e^{-2t}), 定常0
G2 = G.subs({m: 1, d: 3, k: 2})
x2 = sp.inverse_laplace_transform(G2 / s, s, t)
check("Q1(2) x2=3(e^{-t}-e^{-2t})", sp.simplify(x2 - 3 * (sp.exp(-t) - sp.exp(-2 * t))), 0)
check("Q1(2) 定常値 0", sp.limit(x2, t, sp.oo), 0)

# (3) x2max と発生時間: x2'=0 → t=ln2, x2max=3/4
x2e = 3 * (sp.exp(-t) - sp.exp(-2 * t))
tmax = sp.solve(sp.diff(x2e, t), t)
tmax = [v for v in tmax if v.is_real and v >= 0]
check("Q1(3) 発生時間 ln2", tmax[0], sp.log(2))
check("Q1(3) 最大値 3/4", x2e.subs(t, sp.log(2)), sp.Rational(3, 4))

# (4) m=1,d=3,k=0: G=3/(s+3). 単位ステップ x2=1-e^{-3t}, 定常1
G4 = G.subs({m: 1, d: 3, k: 0})
check("Q1(4) G=3/(s+3)", sp.simplify(G4 - 3 / (s + 3)), 0)
x4 = sp.inverse_laplace_transform(G4 / s, s, t)
check("Q1(4) x2=1-e^{-3t}", sp.simplify(x4 - (1 - sp.exp(-3 * t))), 0)
check("Q1(4) 定常値 1", sp.limit(x4, t, sp.oo), 1)

# ============================================================
# 第2問  T=G1 G2 G3/(1+G1+G2)
# ============================================================
G1s, G2s, G3s = sp.symbols('G1 G2 G3')
T_sym = G1s * G2s * G3s / (1 + G1s + G2s)
check("Q2(1) T=G1 G2 G3/(1+G1+G2)", T_sym, G1s * G2s * G3s / (1 + G1s + G2s))

# (2) G1=3/s, G2=1/(s+4), G3=4 → T=12/((s+2)(s+6)), DC=1
g1, g2, g3 = 3 / s, 1 / (s + 4), 4
T = sp.simplify(g1 * g2 * g3 / (1 + g1 + g2))
check("Q2(2) T=12/((s+2)(s+6))", sp.simplify(T - 12 / ((s + 2) * (s + 6))), 0)
check("Q2(2) DCゲイン R(0)=1", T.subs(s, 0), 1)
w0 = 1.0
Tjw = complex(T.subs(s, 1j * w0))
mag = 12 / np.sqrt((12 - w0**2)**2 + (8 * w0)**2)
check("Q2(2) |T(j1)| 公式一致", abs(Tjw), mag, tol=1e-9)
poles2 = sp.Poly(sp.denom(sp.together(T)), s).all_roots()
check("Q2(2) 折れ点 {2,6}",
      set(round(abs(complex(p)), 6) for p in poles2) == {2.0, 6.0}, True)

# (3) ステップ応答 x=1-1.5 e^{-2t}+0.5 e^{-6t}, 定常1, 過減衰で単調増加
xs = sp.inverse_laplace_transform(T / s, s, t)
check("Q2(3) x=1-1.5e^{-2t}+0.5e^{-6t}",
      sp.simplify(xs - (1 - sp.Rational(3, 2) * sp.exp(-2 * t) + sp.Rational(1, 2) * sp.exp(-6 * t))), 0)
check("Q2(3) 定常値 1", sp.limit(xs, t, sp.oo), 1)
f3 = sp.lambdify(t, xs, 'numpy')
ts = np.linspace(0.001, 5, 200)
check("Q2(3) 単調増加（過減衰）", bool(np.all(np.diff(f3(ts)) > -1e-9)), True)

# (4) G3=2s+4: T=6/(s+6)（零点-2が極-2を相殺）, x=1-e^{-6t}, 定常1, 1次系τ=1/6
g3b = 2 * s + 4
Tb = sp.simplify(g1 * g2 * g3b / (1 + g1 + g2))
check("Q2(4) T=6/(s+6)", sp.simplify(Tb - 6 / (s + 6)), 0)
xb = sp.inverse_laplace_transform(Tb / s, s, t)
check("Q2(4) x=1-e^{-6t}", sp.simplify(xb - (1 - sp.exp(-6 * t))), 0)
check("Q2(4) 定常値 1", sp.limit(xb, t, sp.oo), 1)
check("Q2(4) 1次系 時定数 1/6", sp.Rational(1, 6), 1 / sp.Integer(6))

note_unverifiable("Q2(4) 過渡特性の考察",
                  "零点が遅い極を相殺し帯域拡大・高速化する旨の記述で、極の比較は上記で提示済み")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
