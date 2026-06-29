#!/usr/bin/env python3
"""
解答検算テンプレート — 同志社 医工学 院試
======================================
仕様書§4「検証ファースト」: LaTeX を書く前に、これを
  02_solutions/<年度>/verify.py
にコピーして全小問を埋め、★全 PASS にしてから tex を書き始める★。

依存: sympy, mpmath, numpy（制御工学ではscipy.signalもあると便利）
実行: cd 02_solutions/<年度> && python3 verify.py
出力: 小問ごとに [PASS]/[FAIL]/[SKIP]、最後にサマリ。FAIL があれば exit 1。
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30

results = []


def check(label, got, expected, tol=1e-9):
    """記号一致 → だめなら数値一致で PASS/FAIL を判定・記録する。"""
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
    """検算できない小問は理由を残す（証明・図からの読み取り・設計記述など）。
    仕様書§8「検証不能な理由を記録」に対応。"""
    results.append((label, None))
    print(f"[SKIP] {label}: 検証不能 — {reason}")


# === 〔I〕 並列棒 + 剛体棒 ============================
P, L = sp.symbols('P L', positive=True)
A1, E1, A2, E2 = sp.symbols('A1 E1 A2 E2', positive=True)
th = sp.symbols('theta')  # 剛体棒の微小回転角

# 適合: 連結点の鉛直変位は A からの距離に比例
# B は L, C は 3L → lam1 = L*th, lam2 = 3L*th
lam1c = L*th
lam2c = 3*L*th
# 棒の内力 N = AE/(4L) * lam
N1c = A1*E1/(4*L)*lam1c
N2c = A2*E2/(4*L)*lam2c
# A まわりのモーメントつり合い: P*7L = N1*L + N2*3L
sol = sp.solve(sp.Eq(P*7*L, N1c*L + N2c*3*L), th)[0]
N1 = sp.simplify(N1c.subs(th, sol))
N2 = sp.simplify(N2c.subs(th, sol))
D = A1*E1 + 9*A2*E2
check("(I-1) N1", N1, 7*P*A1*E1/D)
check("(I-1) N2", N2, 21*P*A2*E2/D)
# モーメントつり合い確認
check("(I-1) moment balance", sp.simplify(N1*L + N2*3*L), 7*P*L)
lam1 = sp.simplify(N1*4*L/(A1*E1))
lam2 = sp.simplify(N2*4*L/(A2*E2))
check("(I-2) lambda1", lam1, 28*P*L/D)
check("(I-2) lambda2", lam2, 84*P*L/D)
check("(I-2) lambda2 = 3 lambda1", sp.simplify(lam2 - 3*lam1), 0)

# === 〔II〕 段付き片持ちはり ============================
x, f0, b, h, E = sp.symbols('x f0 b h E', positive=True)
# x は固定端 A から、全長 2L、下向き分布荷重 f0
# せん断力（上向き正, 左側和）と曲げモーメント（下に凸正）
Q = f0*(2*L - x)
M = -sp.Rational(1, 2)*f0*(2*L - x)**2
check("(II-1) Q(0)=2f0L", Q.subs(x, 0), 2*f0*L)
check("(II-1) Q(2L)=0", Q.subs(x, 2*L), 0)
check("(II-1) M(0)=-2f0L^2", M.subs(x, 0), -2*f0*L**2)
check("(II-1) M(2L)=0", M.subs(x, 2*L), 0)
# M = -∫Q (dM/dx = Q? check sign): dM/dx should equal Q
check("(II-1) dM/dx = Q", sp.simplify(sp.diff(M, x) - Q), 0)

I1 = b*(2*h)**3/12
I2 = b*h**3/12
check("(II-2) I1", sp.simplify(I1), sp.Rational(2,3)*b*h**3)
check("(II-2) I2", sp.simplify(I2), b*h**3/12)
check("(II-2) I1 = 8 I2", sp.simplify(I1 - 8*I2), 0)

# 最大せん断応力 矩形: tau = 3Q/(2A)
Q1max = Q.subs(x, 0)       # はり①内最大 (x=0)
Q2max = Q.subs(x, L)       # はり②内最大 (x=L)
tau1 = sp.Rational(3,2)*Q1max/(b*2*h)
tau2 = sp.Rational(3,2)*Q2max/(b*h)
check("(II-3) tau1", sp.simplify(tau1), 3*f0*L/(2*b*h))
check("(II-3) tau2", sp.simplify(tau2), 3*f0*L/(2*b*h))

# (4) たわみ: EI v'' = M, 段ごとに I 異なる
C1, C2, C3, C4 = sp.symbols('C1 C2 C3 C4')
v1p = sp.integrate(M/(E*I1), x) + C1
v1 = sp.integrate(v1p, x) + C2
v2p = sp.integrate(M/(E*I2), x) + C3
v2 = sp.integrate(v2p, x) + C4
eqs = [
    v1.subs(x, 0),          # v1(0)=0
    v1p.subs(x, 0),         # v1'(0)=0
    (v1 - v2).subs(x, L),   # 連続
    (v1p - v2p).subs(x, L), # 連続
]
csol = sp.solve(eqs, [C1, C2, C3, C4])
vC = sp.simplify(v2.subs(csol).subs(x, 2*L))
check("(II-4) deflection at C", vC, -23*f0*L**4/(64*E*I2))
check("(II-4) deflection simplified", sp.simplify(vC), sp.simplify(-69*f0*L**4/(16*E*b*h**3)))
# =========================================================


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
