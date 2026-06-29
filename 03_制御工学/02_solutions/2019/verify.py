#!/usr/bin/env python3
"""
AY2019 制御工学 検算
第1問: 壁-バネk-質量m-ダンパd-自由端x1。G=ds/(ms^2+ds+k)。
       ステップ応答x2・最大値と発生時間、指数入力応答と充分時間後の値。
第2問: G1 G2 直列＋K 負帰還。一巡伝達関数のゲイン位相・ボード、閉ループステップ応答、G2変更比較。
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
# (1) m x2''+d x2'+k x2 = d x1' → G=ds/(ms^2+ds+k)
G = d * s / (m * s**2 + d * s + k)
check("Q1(1) 伝達関数 ds/(ms^2+ds+k)", G, d * s / (m * s**2 + d * s + k))

# (2) m=1,d=4,k=3: x2=2(e^{-t}-e^{-3t}), 定常0
G2 = G.subs({m: 1, d: 4, k: 3})
x2 = sp.inverse_laplace_transform(G2 / s, s, t)
check("Q1(2) x2=2(e^{-t}-e^{-3t})", sp.simplify(x2 - 2 * (sp.exp(-t) - sp.exp(-3 * t))), 0)
check("Q1(2) 定常値 0", sp.limit(x2, t, sp.oo), 0)

# (3) 最大値と発生時間: x2'=0 → t=ln3/2, x2max=4√3/9
x2e = 2 * (sp.exp(-t) - sp.exp(-3 * t))
tmax = [v for v in sp.solve(sp.diff(x2e, t), t) if v.is_real and v >= 0]
check("Q1(3) 発生時間 (ln3)/2", tmax[0], sp.log(3) / 2)
check("Q1(3) 最大値 4√3/9",
      sp.simplify(x2e.subs(t, sp.log(3) / 2)), 4 * sp.sqrt(3) / 9)

# (4) m=1,d=2,k=1: G=2s/(s+1)^2, x1=e^{-t} → x2=(2t-t^2)e^{-t}, 充分時間後 0
G4 = G.subs({m: 1, d: 2, k: 1})
check("Q1(4) G=2s/(s+1)^2", sp.simplify(G4 - 2 * s / (s + 1)**2), 0)
X1 = 1 / (s + 1)
x4 = sp.inverse_laplace_transform(G4 * X1, s, t)
check("Q1(4) x2=(2t-t^2)e^{-t}", sp.simplify(x4 - (2 * t - t**2) * sp.exp(-t)), 0)
check("Q1(4) 充分時間後 0", sp.limit(x4, t, sp.oo), 0)

# ============================================================
# 第2問  L=K G1 G2 (一巡), 閉ループ T=G1 G2/(1+K G1 G2)
# ============================================================
G1f = 1 / (s + 1)
G2f = 1 / (10 * s + 1)
K = 2
# (1) 一巡伝達関数 L=2/((s+1)(10s+1))
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
check("Q2(2) DCゲイン[dB]≈6.02", 20 * np.log10(2), 6.0206, tol=1e-3)

# (3) 閉ループ T=1/(10s^2+11s+3)=1/((2s+1)(5s+3)), 極-0.5,-0.6, 定常1/3
T = sp.simplify(G1f * G2f / (1 + K * G1f * G2f))
check("Q2(3) T=1/(10s^2+11s+3)", sp.simplify(T - 1 / (10 * s**2 + 11 * s + 3)), 0)
check("Q2(3) 定常値 1/3", sp.limit(T, s, 0), sp.Rational(1, 3))
xs = sp.inverse_laplace_transform(T / s, s, t)
check("Q2(3) x=1/3-2e^{-t/2}+(5/3)e^{-3t/5}",
      sp.simplify(xs - (sp.Rational(1, 3) - 2 * sp.exp(-t / 2) + sp.Rational(5, 3) * sp.exp(-3 * t / 5))), 0)
check("Q2(3) x(0)=0", xs.subs(t, 0), 0)
polesT = sp.Poly(sp.denom(sp.together(T)), s).all_roots()
check("Q2(3) 閉ループ極 {-0.5,-0.6}",
      set(sp.nsimplify(p) for p in polesT) == {sp.Rational(-1, 2), sp.Rational(-3, 5)}, True)

# (4) G2=1/(0.1s+1): T=10/((s+5)(s+6)), 定常1/3, 極-5,-6（高速化）
G2b = 1 / (sp.Rational(1, 10) * s + 1)
Tb = sp.simplify(G1f * G2b / (1 + K * G1f * G2b))
check("Q2(4) T=10/((s+5)(s+6))", sp.simplify(Tb - 10 / ((s + 5) * (s + 6))), 0)
check("Q2(4) 定常値 1/3", sp.limit(Tb, s, 0), sp.Rational(1, 3))
xb = sp.inverse_laplace_transform(Tb / s, s, t)
check("Q2(4) x=1/3-2e^{-5t}+(5/3)e^{-6t}",
      sp.simplify(xb - (sp.Rational(1, 3) - 2 * sp.exp(-5 * t) + sp.Rational(5, 3) * sp.exp(-6 * t))), 0)
polesTb = sp.Poly(sp.denom(sp.together(Tb)), s).all_roots()
check("Q2(4) 閉ループ極 {-5,-6}（(3)より高速）",
      set(sp.nsimplify(p) for p in polesTb) == {sp.Integer(-5), sp.Integer(-6)}, True)

note_unverifiable("Q2(4) 2系の違いの考察",
                  "G2 の時定数短縮で帯域拡大・応答高速化（定常値は同じ1/3）の記述。極比較で根拠提示済み")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
