#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2022
=========================================
〔I〕  sinh(x^2+y) のマクローリン展開（3次まで）
〔II〕 ∭_D y^2 dV, D: 球 x^2+y^2+z^2<=a^2
〔III〕4ベクトルの1次独立性と Gram-Schmidt
〔IV〕 対角化可能性
実行: cd 02_solutions/2022 && python3 verify.py
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


# === 〔I〕 sinh(x^2+y) 展開 ======================================
x, y = sp.symbols('x y')
e1 = sp.sinh(x**2 + y)
s = e1.series(x, 0, 4).removeO().series(y, 0, 4).removeO()
poly = sp.Poly(sp.expand(s), x, y)
trunc = sum(co*x**m*y**n for (m, n), co in poly.terms() if m + n <= 3)
expected1 = y + x**2 + y**3/6
check("(I) 3次まで: y + x^2 + y^3/6", sp.expand(trunc), sp.expand(expected1))

# === 〔II〕 ∭ y^2 over ball =====================================
a = sp.symbols('a', positive=True)
rho, th, ph = sp.symbols('rho theta phi', positive=True)
# y = rho sinθ sinφ
integrand = (rho*sp.sin(th)*sp.sin(ph))**2 * rho**2*sp.sin(th)
V = sp.integrate(sp.integrate(sp.integrate(integrand, (rho, 0, a)),
                              (th, 0, sp.pi)), (ph, 0, 2*sp.pi))
check("(II) ∭ y^2 = 4πa^5/15", sp.simplify(V), 4*sp.pi*a**5/15)

# === 〔III〕 1次独立 + Gram-Schmidt =============================
v1 = sp.Matrix([1, 0, 1, 0]); v2 = sp.Matrix([0, 1, 1, 0])
v3 = sp.Matrix([2, 0, 0, 1]); v4 = sp.Matrix([1, 1, 1, 1])
Mv = sp.Matrix.hstack(v1, v2, v3, v4)
check("(III) 1次独立 (det≠0)", Mv.det() != 0, True)
e1v = sp.Matrix([1, 0, 1, 0])/sp.sqrt(2)
e2v = sp.Matrix([-1, 2, 1, 0])/sp.sqrt(6)
e3v = sp.Matrix([2, 2, -2, 3])/sp.sqrt(21)
e4v = sp.Matrix([-1, -1, 1, 2])/sp.sqrt(7)
es = [e1v, e2v, e3v, e4v]
for i, ev in enumerate(es, 1):
    check(f"(III) |e{i}|=1", sp.simplify(ev.dot(ev)), 1)
for i in range(4):
    for j in range(i+1, 4):
        check(f"(III) e{i+1}·e{j+1}=0", sp.simplify(es[i].dot(es[j])), 0)
gs = sp.GramSchmidt([v1, v2, v3, v4], orthonormal=True)
check("(III) sympy GS と一致(符号許容)",
      all(sp.simplify((gs[i]-es[i]).norm()) == 0 or sp.simplify((gs[i]+es[i]).norm()) == 0
          for i in range(4)), True)

# === 〔IV〕 対角化可能性 ========================================
a_, b_ = sp.symbols('a b')
A = sp.Matrix([[a_, 1, 0], [0, a_, 0], [0, 0, b_]])
# a≠b の一般の場合（具体値 a=2,b=5 で確認）
A2 = A.subs({a_: 2, b_: 5})
check("(IV) a≠b で対角化不可", A2.is_diagonalizable(), False)
# 固有値 a の幾何的重複度
geo = 3 - (A2 - 2*sp.eye(3)).rank()
check("(IV) λ=a の幾何的重複度=1(<代数2)", geo, 1)
# a=b の場合も不可
A3 = A.subs({a_: 3, b_: 3})
check("(IV) a=b でも対角化不可", A3.is_diagonalizable(), False)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
