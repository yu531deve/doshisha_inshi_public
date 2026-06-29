#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2018
=========================================
〔I〕  4x4 行列式の因数分解（空欄 ア,イ）
〔II〕 Ax=b（|A|=ウ, x3=(1/|A|)エ, 固有値オ）
〔III〕∬_D xy dxdy, D: |x-1|+y<=1, y>=0
〔IV〕 1/(1+x+y^2) のマクローリン（カ,キ,ク）
実行: cd 02_solutions/2018 && python3 verify.py
"""
import sympy as sp

results = []


def check(label, got, expected, tol=1e-9):
    ok = False
    try:
        if isinstance(got, bool) or isinstance(expected, bool):
            ok = bool(got) == bool(expected)
        else:
            ok = sp.simplify(sp.sympify(got) - sp.sympify(expected)) == 0
    except Exception:
        pass
    if not ok:
        try:
            ok = bool(got == expected)
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


a, b, c = sp.symbols('a b c')

# === 〔I〕 行列式の因数分解 =======================================
M = sp.Matrix([[0, a, b, c], [a, 0, c, b], [b, c, 0, a], [c, b, a, 0]])
detM = sp.factor(M.det())
print("det =", detM)
# 確定: (a+b+c)(a-b-c)(a-b+c)(a+b-c). ア=a-b+c, イ=a+b-c
expected = (a+b+c)*(a-b-c)*(a-b+c)*(a+b-c)
check("(I) det = (a+b+c)(a-b-c)(a-b+c)(a+b-c)", sp.expand(M.det()), sp.expand(expected))
print("→ ア=a-b+c, イ=a+b-c")

# === 〔II〕 Ax=b ================================================
A = sp.Matrix([[5, 1, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 1, 2]])
b1, b2, b3, b4 = sp.symbols('b1 b2 b3 b4')
bb = sp.Matrix([b1, b2, b3, b4])
check("(II-ウ) |A| = 30", A.det(), 30)
# Cramer で x3
A3 = A.copy(); A3[:, 2] = bb
x3 = sp.simplify(A3.det()/A.det())
check("(II-エ) x3*|A| = 20 b3 - 10 b2", sp.expand(A3.det()), sp.expand(20*b3 - 10*b2))
# 直接解とも照合
xsol = A.solve(bb)
check("(II) x3 一致", sp.simplify(xsol[2] - x3), 0)
eig = A.eigenvals()
print("eigenvals =", eig)
check("(II-オ) 固有値集合 {5,1,2,3}",
      set(eig.keys()) == {sp.Integer(5), sp.Integer(1), sp.Integer(2), sp.Integer(3)}, True)

# === 〔III〕 ∬ xy ===============================================
x, y = sp.symbols('x y', real=True)
val3 = sp.integrate(sp.integrate(x*y, (x, y, 2 - y)), (y, 0, 1))
check("(III) ∬_D xy = 1/3", val3, sp.Rational(1, 3))

# === 〔IV〕 1/(1+x+y^2) 展開 =====================================
f = 1/(1 + x + y**2)
s = f.series(x, 0, 3).removeO().series(y, 0, 3).removeO()
poly = sp.Poly(sp.expand(s), x, y)
trunc = sum(co*x**m*y**n for (m, n), co in poly.terms() if m + n <= 2)
expected4 = 1 - x + x**2 - y**2
check("(IV) 2次まで: 1 - x + x^2 - y^2", sp.expand(trunc), sp.expand(expected4))
print("→ カ=-1, キ=1, ク=-1")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
