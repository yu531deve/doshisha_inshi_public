#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2023
=========================================
〔I〕  e^{ix}=cosx+i sinx（マクローリンで証明）
〔II〕 ∬_D arctan(y/x) dxdy（四分円）
〔III〕4x4 行列の対角化（a≠3）
〔IV〕 同次系が非自明解を持つ λ と最小 λ の解
実行: cd 02_solutions/2023 && python3 verify.py
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


x = sp.symbols('x', real=True)

# === 〔I〕 オイラーの公式（級数で確認） ==========================
N = 12
e_ix = sum((sp.I*x)**n/sp.factorial(n) for n in range(N))
cs = sum((-1)**k*x**(2*k)/sp.factorial(2*k) for k in range(N//2))
sn = sum((-1)**k*x**(2*k+1)/sp.factorial(2*k+1) for k in range(N//2))
check("(I) Σ(ix)^n/n! = cos+isin (部分和一致)",
      sp.expand(e_ix - (cs + sp.I*sn)), 0)
# 厳密にも
check("(I) e^{ix} = cosx + i sinx", sp.simplify(sp.exp(sp.I*x) - (sp.cos(x)+sp.I*sp.sin(x))), 0)

# === 〔II〕 ∬ arctan(y/x) =======================================
r, th = sp.symbols('r theta', positive=True)
# 極座標で arctan(y/x)=θ
val = sp.integrate(sp.integrate(th*r, (r, 0, sp.sqrt(2))), (th, 0, sp.pi/2))
check("(II) ∬ arctan(y/x) = π^2/8", sp.simplify(val), sp.pi**2/8)

# === 〔III〕 対角化（a≠3） ======================================
a_, lam = sp.symbols('a lambda')
A = sp.Matrix([[a_, 0, 0, 0], [0, 1, 2, 0], [0, 0, 2, 0], [1, 0, 0, 3]])
cp = sp.factor(A.charpoly(lam).as_expr())
print("char poly =", cp)
check("(III) 固有値 a,1,2,3",
      sp.expand(A.charpoly(lam).as_expr()),
      sp.expand((a_-lam)*(1-lam)*(2-lam)*(3-lam)))
# 具体値 a=5(≠3,1,2) で対角化可能、D=diag
A5 = A.subs(a_, 5)
check("(III) a=5 で対角化可能", A5.is_diagonalizable(), True)
# a=1（重複）でも可能か
check("(III) a=1 でも対角化可能", A.subs(a_, 1).is_diagonalizable(), True)
check("(III) a=2 でも対角化可能", A.subs(a_, 2).is_diagonalizable(), True)

# === 〔IV〕 同次系 ==============================================
xv, yv, zv = sp.symbols('x y z')
M = sp.Matrix([[lam+1, 1, 4], [8, lam, 10], [-4, -1, lam-7]])
dM = sp.factor(M.det())
print("det M =", dM)
check("(IV) det = (λ-1)(λ-2)(λ-3)",
      sp.expand(M.det()), sp.expand((lam-1)*(lam-2)*(lam-3)))
# 最小 λ=1 の解
M1 = M.subs(lam, 1)
ns = M1.nullspace()
print("nullspace(λ=1) =", [v.T.tolist() for v in ns])
check("(IV) λ=1 の解は1次元", len(ns), 1)
sol = sp.Matrix([-1, -2, 1])
check("(IV) (-1,-2,1) が解", M1*sol == sp.zeros(3, 1), True)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
