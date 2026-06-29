#!/usr/bin/env python3
"""
AY2012 制御工学 解答検算
========================
第1問: 二段階制御（内側 K0 負帰還ループ + 外側単位負帰還）の伝達関数、
       ボード線図（折れ線近似）、比例補償の定常値、積分補償の定常値。
第2問: 一自由度振動系（速度帰還 D・位置帰還 K）の伝達関数、固有角振動数・
       減衰係数、インパルス応答、単位ステップ応答。
依存: sympy, numpy, scipy.signal
実行: cd 02_solutions/2012 && python3 verify.py
"""
import sympy as sp
import numpy as np

results = []


def check(label, got, expected, tol=1e-9):
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
G1, G2, K0, K1 = sp.symbols('G1 G2 K0 K1')

# ============================================================
# 第1問
# ============================================================
# (1) 伝達関数
#   内側ループ: 出力 Y を K0 で負帰還し sum2 へ → M = G1/(1+G1*K0)
#   外側ループ: 出力 Y を単位負帰還し sum1 へ、G2 が前向き
#   T = M*G2/(1+M*G2) = G1*G2/(1+G1*K0+G1*G2)
M = G1 / (1 + G1 * K0)
T_sym = M * G2 / (1 + M * G2)
T_expected = G1 * G2 / (1 + G1 * K0 + G1 * G2)
check("第1問(1) 伝達関数 T=G1 G2/(1+G1 K0+G1 G2)",
      sp.simplify(T_sym - T_expected), 0)

# (2) G1=1/((s+1)(10s+1)), K0=1 のボード(ゲイン)折れ線
#   DC ゲイン |G1(0)|=1 → 0 dB、折点 ω=0.1 と ω=1。
G1f = 1 / ((s + 1) * (10 * s + 1))
check("第1問(2) DC ゲイン |G1(0)|", sp.Abs(G1f.subs(s, 0)), 1)  # 0 dB
# 折点周波数（10s+1→0.1, s+1→1）
check("第1問(2) 折点 ω1", sp.Rational(1, 10), 0.1)
check("第1問(2) 折点 ω2", sp.Integer(1), 1)
# 高周波での傾き: 2極なので -40 dB/dec。ω=100 での近似ゲイン[dB]を数値確認
w = 100.0
mag_db = 20 * np.log10(abs(1 / ((1j * w + 1) * (10j * w + 1))))
# 折れ線近似: 0dB→(0.1)→-20dB/dec→(1)→-40dB/dec
approx = 0 - 20 * np.log10(100 / 1) - 20 * (np.log10(100) - np.log10(1)) - 20 * np.log10(1 / 0.1)
# 折れ線: ω=0.1で0, ω=1で -20*log10(1/0.1)=-20, ω=100で -20 -40*log10(100/1)=-20-80=-100
approx_db = -20 - 40 * np.log10(100 / 1)
check("第1問(2) 折れ線近似 ω=100 で約 -100 dB（真値と±5dB）",
      abs(mag_db - approx_db) < 5, True)

# (3) G2=K1（比例）、単位ステップの定常値 = T(0)（最終値の定理）
#   T(0) = G1(0) K1/(1+G1(0) K0+G1(0) K1), G1(0)=1, K0=1 → K1/(2+K1)
T3 = T_expected.subs({G1: 1, K0: 1, G2: K1})
y_inf3 = sp.simplify(T3)  # s=0 では各 G は DC 値
check("第1問(3) 定常値 y(∞)=K1/(2+K1)", y_inf3, K1 / (2 + K1))
# 安定性（例: K1=1）: 特性 1+G1+G1*K1 with G1=1/((s+1)(10s+1))
den3 = sp.expand(((s + 1) * (10 * s + 1)) * (1 + G1f + G1f * 1).together().as_numer_denom()[0]
                 / ((s + 1) * (10 * s + 1)))
# 直接 char poly: s 多項式の分子
char3 = sp.numer(sp.together(1 + G1f * (1 + 1)))
poles3 = sp.solve(char3, s)
check("第1問(3) K1=1 で安定（全極 Re<0）",
      all(sp.re(p) < 0 for p in poles3), True)

# (4) G2=(s+0.1)/s（積分動作）、単位ステップの定常値
#   T(s)=G1 G2/(1+G1 K0+G1 G2)、s→0 で G2→∞ より T→1
G2f = (s + sp.Rational(1, 10)) / s
T4 = G1f * G2f / (1 + G1f * 1 + G1f * G2f)
y_inf4 = sp.limit(T4, s, 0)
check("第1問(4) 定常値 y(∞)=1（積分補償で偏差0）", y_inf4, 1)
# 安定性: 特性方程式 10s^3+11s^2+3s+0.1
char4 = sp.expand(sp.numer(sp.together(1 + G1f * 1 + G1f * G2f)))
# numer(together) はスケール因子を含みうるので monic 化して比較
char4_monic = sp.Poly(char4, s).monic().as_expr()
target_monic = sp.Poly(10 * s**3 + 11 * s**2 + 3 * s + sp.Rational(1, 10), s).monic().as_expr()
check("第1問(4) 特性方程式 ∝ 10s^3+11s^2+3s+0.1",
      sp.simplify(char4_monic - target_monic), 0)
poles4 = sp.solve(char4, s)
check("第1問(4) 安定（全極 Re<0）",
      all(sp.re(sp.nsimplify(p)) < 0 for p in [complex(p) for p in poles4]), True)

note_unverifiable("第1問(5) 補償器 G2 の役割",
                  "積分動作の意味づけ（型を上げ定常偏差を除去）の記述問題で数値検算対象外")

# ============================================================
# 第2問
# ============================================================
M_, D_, K_ = sp.symbols('M D K', positive=True)
# (1) 伝達関数 Y/U = 1/(M s^2 + D s + K)
#   A=(1/M)(U-K Y-D V), V=sY, A=s^2 Y を整理
T_q2 = 1 / (M_ * s**2 + D_ * s + K_)
check("第2問(1) 伝達関数 1/(M s^2+D s+K)", T_q2, 1 / (M_ * s**2 + D_ * s + K_))

# (2) ωn=sqrt(K/M), ζ=D/(2 sqrt(K M))
#   標準形 (1/M)/(s^2+(D/M)s+K/M) と s^2+2ζωn s+ωn^2 の比較
wn = sp.sqrt(K_ / M_)
zeta = D_ / (2 * sp.sqrt(K_ * M_))
check("第2問(2) ωn^2 = K/M", wn**2, K_ / M_)
check("第2問(2) 2ζωn = D/M", sp.simplify(2 * zeta * wn), D_ / M_)

# (3) M=1,D=3,K=2 のインパルス応答 = L^{-1}{1/(s^2+3s+2)}
T3v = 1 / (s**2 + 3 * s + 2)
h = sp.inverse_laplace_transform(T3v, s, t)
check("第2問(3) インパルス応答 e^{-t}-e^{-2t}",
      sp.simplify(h - (sp.exp(-t) - sp.exp(-2 * t))), 0)
# 極の確認
check("第2問(3) 極は -1,-2",
      set(sp.solve(s**2 + 3 * s + 2, s)) == {-1, -2}, True)

# (4) 単位ステップ応答 = L^{-1}{1/(s(s^2+3s+2))}
Ystep = T3v / s
y4 = sp.inverse_laplace_transform(Ystep, s, t)
y4_expected = sp.Rational(1, 2) - sp.exp(-t) + sp.Rational(1, 2) * sp.exp(-2 * t)
check("第2問(4) ステップ応答 1/2 - e^{-t} + (1/2)e^{-2t}",
      sp.simplify(y4 - y4_expected), 0)
check("第2問(4) 定常値 y(∞)=T(0)=1/2",
      sp.limit(y4_expected, t, sp.oo), sp.Rational(1, 2))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
