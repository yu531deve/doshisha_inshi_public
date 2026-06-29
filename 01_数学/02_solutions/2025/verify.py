#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2025
=========================================
〔I〕 sin x の x=0 近傍 Taylor 展開
〔II〕 lim_{n→∞} ∬_{D_n} e^{-x^2-y^2} dxdy, D_n: 第1象限の四分円
〔III〕 A の逆行列
〔IV〕 B の対角化
実行: cd 02_solutions/2025 && python3 verify.py
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30

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


# === 〔I〕 sin x の Taylor 展開 =====================================
x = sp.symbols('x')
# 確定解答: sin x = Σ_{k=0}^∞ (-1)^k x^{2k+1}/(2k+1)!
k = sp.symbols('k', integer=True, nonnegative=True)
series_sum = sp.Sum((-1)**k * x**(2*k+1) / sp.factorial(2*k+1), (k, 0, sp.oo))
# 低次の係数比較（series で照合）
taylor = sp.series(sp.sin(x), x, 0, 10).removeO()
expected_taylor = x - x**3/6 + x**5/120 - x**7/5040 + x**9/362880
check("(I) Taylor 多項式 (9次まで)", taylor, expected_taylor)
# 一般項の正しさ: 部分和(0..4) が series と一致
partial = sum((-1)**i * x**(2*i+1) / sp.factorial(2*i+1) for i in range(5))
check("(I) 一般項の部分和=展開", sp.expand(partial), sp.expand(expected_taylor))
# 収束半径 R=∞ の確認: 比判定 |a_{k+1}/a_k| → 0
ratio_limit = sp.limit(sp.factorial(2*k+1)/sp.factorial(2*k+3), k, sp.oo)
check("(I) 収束半径無限大 (比→0)", ratio_limit, 0)

# === 〔II〕 ∬ e^{-x^2-y^2} 第1象限四分円, n→∞ =====================
# 極座標: ∫_0^{π/2}∫_0^∞ e^{-r^2} r dr dθ = (π/2)(1/2) = π/4
r, th = sp.symbols('r theta', positive=True)
val = sp.integrate(sp.exp(-r**2)*r, (r, 0, sp.oo)) * sp.integrate(1, (th, 0, sp.pi/2))
check("(II) 広義二重積分 (第1象限全体)", val, sp.pi/4)
# 数値照合: 有限 n=8 で四分円上を数値積分し π/4 に近いこと
num = mp.quad(lambda rr: mp.e**(-rr**2)*rr, [0, 8]) * (mp.pi/2)
check("(II) 数値照合 (n=8)", num, mp.pi/4, tol=1e-6)

# === 〔III〕 A の逆行列 ============================================
A = sp.Matrix([[1, 2, 1], [0, 1, 0], [2, -1, 1]])
Ainv = A.inv()
print("A^{-1} =", Ainv.tolist())
# det A
check("(III) det A", A.det(), -1)
# 確定解答（手計算結果）と照合
Ainv_expected = sp.Matrix([[-1, 3, 1], [0, 1, 0], [2, -5, -1]])
check("(III) 逆行列が手計算と一致", sp.simplify(Ainv - Ainv_expected) == sp.zeros(3, 3), True)
check("(III) A * A^{-1} = I", sp.simplify(A*Ainv_expected) == sp.eye(3), True)

# === 〔IV〕 B の対角化 ============================================
B = sp.Matrix([[4, 1, 0], [0, 2, 0], [0, 0, 3]])
# 固有値（上三角なので対角成分）
eig = B.eigenvals()
print("B eigenvals =", eig)
check("(IV) 固有値の集合={2,3,4}", set(eig.keys()) == {sp.Integer(4), sp.Integer(2), sp.Integer(3)}, True)
check("(IV) 各固有値は単根", all(m == 1 for m in eig.values()), True)
# 対角化可能（相異なる3固有値）→ P, D を構成
P, D = B.diagonalize()
print("P =", P.tolist())
print("D =", D.tolist())
check("(IV) P^{-1} B P = D", sp.simplify(P.inv()*B*P) == D, True)
check("(IV) D の対角成分集合", set([D[i, i] for i in range(3)]) == {sp.Integer(2), sp.Integer(3), sp.Integer(4)}, True)
# 確定解答の P（固有ベクトル: λ=4→(1,0,0), λ=2→(1,-2,0), λ=3→(0,0,1)）で検算
P_exp = sp.Matrix([[1, 1, 0], [0, -2, 0], [0, 0, 1]])
D_exp = sp.diag(4, 2, 3)
check("(IV) 確定 P,D で P^{-1}BP=D", sp.simplify(P_exp.inv()*B*P_exp) == D_exp, True)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
