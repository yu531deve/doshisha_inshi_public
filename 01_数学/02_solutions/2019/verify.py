#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2019
=========================================
〔I〕  ∫ e^x sin x dx
〔II〕 ∭_{R^3} e^{-(x^2+y^2+z^2)} dV
〔III〕|A| = |A^t| の証明
〔IV〕 x軸まわりの回転行列と等長性
実行: cd 02_solutions/2019 && python3 verify.py
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


x, y, z, th = sp.symbols('x y z theta', real=True)

# === 〔I〕 ∫ e^x sin x dx =======================================
F = sp.integrate(sp.exp(x)*sp.sin(x), x)
print("∫ =", sp.simplify(F))
expected1 = sp.exp(x)*(sp.sin(x) - sp.cos(x))/2
# 不定積分は定数差を許すので、微分して被積分関数に戻るかで確認
check("(I) d/dx[e^x(sinx-cosx)/2] = e^x sinx",
      sp.simplify(sp.diff(expected1, x) - sp.exp(x)*sp.sin(x)), 0)
check("(I) sympy 積分と定数差", sp.simplify(sp.diff(F - expected1, x)), 0)

# === 〔II〕 ∭ e^{-(x^2+y^2+z^2)} ================================
I1 = sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))
check("(II) 1次元 = √π", I1, sp.sqrt(sp.pi))
check("(II) 3重積分 = π^{3/2}", I1**3, sp.pi**sp.Rational(3, 2))

# === 〔III〕 |A| = |A^t|（具体行列で確認） ========================
for n in (2, 3, 4):
    Arand = sp.Matrix(n, n, lambda i, j: sp.Symbol(f'a_{i}{j}'))
    check(f"(III) |A|=|A^t| (n={n}, 記号)", sp.simplify(Arand.det() - Arand.T.det()), 0)

# === 〔IV〕 回転行列と等長性 ====================================
R = sp.Matrix([[1, 0, 0],
               [0, sp.cos(th), -sp.sin(th)],
               [0, sp.sin(th), sp.cos(th)]])
# 等長変換: R^t R = I, det R = 1
check("(IV) R^t R = I", sp.simplify(R.T*R) == sp.eye(3), True)
check("(IV) det R = 1", sp.simplify(R.det()), 1)
# ノルム保存: |Rv| = |v| を記号ベクトルで確認
v = sp.Matrix(sp.symbols('v1 v2 v3', real=True))
check("(IV) |Rv|^2 = |v|^2", sp.simplify((R*v).dot(R*v) - v.dot(v)), 0)
# x軸上の点は不変
check("(IV) x軸は不変", sp.simplify(R*sp.Matrix([1, 0, 0]) - sp.Matrix([1, 0, 0])) == sp.zeros(3, 1), True)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
