#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2012
=========================================
〔I〕  f=log(x^2+y^2) のラプラシアン
〔II〕 ∬_D (x^2+2xy-4y^3) dxdy, D=[-1,1]×[0,1]
〔III〕∫_{-∞}^{∞} e^{-x^2/2} dx
〔IV〕 3x3 行列式
〔V〕  同行列の逆行列
〔VI〕 3x3 行列の固有値
実行: cd 02_solutions/2012 && python3 verify.py
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


# === 〔I〕 ラプラシアン =============================================
x, y = sp.symbols('x y', real=True)
f = sp.log(x**2 + y**2)
lap = sp.diff(f, x, 2) + sp.diff(f, y, 2)
check("(I) Δf = 0 (原点を除く)", sp.simplify(lap), 0)

# === 〔II〕 重積分 ==================================================
val2 = sp.integrate(sp.integrate(x**2 + 2*x*y - 4*y**3, (x, -1, 1)), (y, 0, 1))
check("(II) ∬_D", val2, sp.Rational(-4, 3))

# === 〔III〕 ガウス積分 =============================================
val3 = sp.integrate(sp.exp(-x**2/2), (x, -sp.oo, sp.oo))
check("(III) ∫ e^{-x^2/2} dx", val3, sp.sqrt(2*sp.pi))

# === 〔IV〕 行列式 =================================================
M = sp.Matrix([[1, 3, 1], [0, 1, 1], [1, 3, 2]])
check("(IV) det M", M.det(), 1)

# === 〔V〕 逆行列 ==================================================
Minv = M.inv()
print("M^{-1} =", Minv.tolist())
Minv_exp = sp.Matrix([[-1, -3, 2], [1, 1, -1], [-1, 0, 1]])
check("(V) 逆行列が手計算と一致", sp.simplify(Minv - Minv_exp) == sp.zeros(3, 3), True)
check("(V) M*M^{-1}=I", sp.simplify(M*Minv_exp) == sp.eye(3), True)

# === 〔VI〕 固有値 =================================================
N = sp.Matrix([[-1, -1, -4], [-8, 0, -10], [4, 1, 7]])
eig = N.eigenvals()
print("N eigenvals =", eig)
check("(VI) 固有値の集合={1,2,3}",
      set(eig.keys()) == {sp.Integer(1), sp.Integer(2), sp.Integer(3)}, True)
check("(VI) char poly = λ^3-6λ^2+11λ-6",
      sp.expand(N.charpoly().as_expr()),
      sp.expand((sp.Symbol('lambda')-1)*(sp.Symbol('lambda')-2)*(sp.Symbol('lambda')-3)))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
