#!/usr/bin/env python3
"""
AY2025 制御工学 検算
第1問: 壁-(k2∥d2)-質量m-(k1∥d1)-入力端A(x1)。G=(d1 s+k1)/(ms^2+(d1+d2)s+(k1+k2))。
       定常値・ステップ応答・最大値、(5)A固定で外力ステップ。
第2問: 図2=Gc Gp 単位帰還。Gc=2/(s+3)（比例/ラグ）と Gc=(s+10)/s（積分=PI）の
       ステップ応答・一巡ボードを比較。
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
# 第1問  G=(d1 s+k1)/(m s^2+(d1+d2)s+(k1+k2))
# ============================================================
m, d1, d2, k1, k2 = sp.symbols('m d1 d2 k1 k2', positive=True)
G = (d1 * s + k1) / (m * s**2 + (d1 + d2) * s + (k1 + k2))

# (1) 定常値 = G(0) = k1/(k1+k2)
check("Q1(1) 定常値 k1/(k1+k2)", sp.limit(G, s, 0), k1 / (k1 + k2))

# (2) m=1,d1=1,d2=1,k1=1,k2=2 → G=(s+1)/(s^2+2s+3), 定常1/3
G2 = G.subs({m: 1, d1: 1, d2: 1, k1: 1, k2: 2})
check("Q1(2) G=(s+1)/(s^2+2s+3)", sp.simplify(G2 - (s + 1) / (s**2 + 2 * s + 3)), 0)
x2 = sp.inverse_laplace_transform(G2 / s, s, t)
check("Q1(2) x2=1/3+(1/3)e^{-t}(√2 sin√2 t - cos√2 t)",
      sp.simplify(x2 - (sp.Rational(1, 3) + sp.Rational(1, 3) * sp.exp(-t) * (sp.sqrt(2) * sp.sin(sp.sqrt(2) * t) - sp.cos(sp.sqrt(2) * t)))), 0)
check("Q1(2) 定常値 1/3", sp.limit(x2, t, sp.oo), sp.Rational(1, 3))

# (3) 最大値: x2'=e^{-t}cos(√2 t)=0 → 最初のピーク √2 t=π/2 → t=π/(2√2)
dx = sp.simplify(sp.diff(x2, t))
check("Q1(3) dx/dt = e^{-t}cos(√2 t)", sp.simplify(dx - sp.exp(-t) * sp.cos(sp.sqrt(2) * t)), 0)
tmax = sp.pi / (2 * sp.sqrt(2))
x2max = sp.simplify(x2.subs(t, tmax))
print("  [info] Q1(3) x2max =", x2max, "=", float(x2max), " at t=π/(2√2)=", float(tmax))
check("Q1(3) x2max=(1/3)(1+√2 e^{-π/(2√2)})",
      sp.simplify(x2max - sp.Rational(1, 3) * (1 + sp.sqrt(2) * sp.exp(-sp.pi / (2 * sp.sqrt(2))))), 0)

# (4) m=1,d1=3,d2=0,k1=1,k2=1 → G=(3s+1)/((s+1)(s+2)), 定常1/2
G4 = G.subs({m: 1, d1: 3, d2: 0, k1: 1, k2: 1})
check("Q1(4) G=(3s+1)/(s^2+3s+2)", sp.simplify(G4 - (3 * s + 1) / (s**2 + 3 * s + 2)), 0)
x4 = sp.inverse_laplace_transform(G4 / s, s, t)
check("Q1(4) x2=1/2+2e^{-t}-(5/2)e^{-2t}",
      sp.simplify(x4 - (sp.Rational(1, 2) + 2 * sp.exp(-t) - sp.Rational(5, 2) * sp.exp(-2 * t))), 0)
check("Q1(4) 定常値 1/2", sp.limit(x4, t, sp.oo), sp.Rational(1, 2))
# 最大値: x2'=-2e^{-t}+5e^{-2t}=0 → e^t=5/2 → t=ln(5/2), x2max=9/10
tm4 = [v for v in sp.solve(sp.diff(x4, t), t) if v.is_real]
check("Q1(4) 発生時間 ln(5/2)", tm4[0], sp.log(sp.Rational(5, 2)))
check("Q1(4) 最大値 9/10", sp.simplify(x4.subs(t, sp.log(sp.Rational(5, 2)))), sp.Rational(9, 10))

# (5) A固定, m=1,d1=1,d2=1,k1=1,k2=1 → x2''+2x2'+2x2=f, 定常1/2
G5 = 1 / (s**2 + 2 * s + 2)
x5 = sp.inverse_laplace_transform(G5 / s, s, t)
check("Q1(5) x2=1/2-(1/2)e^{-t}(cos t+sin t)",
      sp.simplify(x5 - (sp.Rational(1, 2) - sp.Rational(1, 2) * sp.exp(-t) * (sp.cos(t) + sp.sin(t)))), 0)
check("Q1(5) 定常値 1/2", sp.limit(x5, t, sp.oo), sp.Rational(1, 2))

# ============================================================
# 第2問  単位帰還 forward Gc Gp
# ============================================================
Gp = 1 / (s + 1)
# (1) Gc=2/(s+3): T=2/(s^2+4s+5), y=2/5-(2/5)e^{-2t}(cos t+2 sin t), 定常2/5
Gc1 = 2 / (s + 3)
T1 = sp.simplify(Gc1 * Gp / (1 + Gc1 * Gp))
check("Q2(1) T=2/(s^2+4s+5)", sp.simplify(T1 - 2 / (s**2 + 4 * s + 5)), 0)
y1 = sp.inverse_laplace_transform(T1 / s, s, t)
check("Q2(1) y=2/5-(2/5)e^{-2t}(cos t+2 sin t)",
      sp.simplify(y1 - (sp.Rational(2, 5) - sp.Rational(2, 5) * sp.exp(-2 * t) * (sp.cos(t) + 2 * sp.sin(t)))), 0)
check("Q2(1) 定常値 2/5", sp.limit(y1, t, sp.oo), sp.Rational(2, 5))
L1 = sp.simplify(Gc1 * Gp)
check("Q2(1) 一巡DCゲイン L(0)=2/3", L1.subs(s, 0), sp.Rational(2, 3))
check("Q2(1) 一巡DC[dB]=-3.52", 20 * np.log10(2 / 3), -3.52, tol=1e-2)

# (2) Gc=(s+10)/s: T=(s+10)/(s^2+2s+10), y=1-e^{-t}cos3t, 定常1（積分→偏差0）
Gc2 = (s + 10) / s
T2 = sp.simplify(Gc2 * Gp / (1 + Gc2 * Gp))
check("Q2(2) T=(s+10)/(s^2+2s+10)", sp.simplify(T2 - (s + 10) / (s**2 + 2 * s + 10)), 0)
y2 = sp.inverse_laplace_transform(T2 / s, s, t)
check("Q2(2) y=1-e^{-t}cos3t", sp.simplify(y2 - (1 - sp.exp(-t) * sp.cos(3 * t))), 0)
check("Q2(2) 定常値 1（偏差0）", sp.limit(y2, t, sp.oo), 1)
# 一巡 L=(s+10)/(s(s+1)) は積分器をもつ（低周波 ∝10/ω, ω=10で0dB）
L2 = sp.simplify(Gc2 * Gp)
check("Q2(2) 一巡 L=(s+10)/(s(s+1))", sp.simplify(L2 - (s + 10) / (s * (s + 1))), 0)
check("Q2(2) 低周波 |L|≈10/ω (ω=10→0dB)",
      20 * np.log10(abs(complex(L2.subs(s, 1j * 0.001)))), 20 * np.log10(10 / 0.001), tol=0.2)

note_unverifiable("Q2(3) Gc の働きの違いの考察",
                  "(1)比例/ラグ→DC有限ゲイン・定常偏差3/5、(2)積分(PI)→type上昇・偏差0で追従。数値根拠は定常値2/5 vs 1, DC -3.52dB vs ∞")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
