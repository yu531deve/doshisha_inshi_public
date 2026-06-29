#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2016
=========================================
〔I〕  4x4 行列式
〔II〕 同行列の逆行列
〔III〕同行列の固有値
〔IV〕 線型写像の rank と dim ker
〔V〕  ∬_D 1/(1+x^2+y^2)^2 dxdy（レムニスケート領域）
〔VI〕 球座標のヤコビアン
〔VII〕sinh(x+y) のマクローリン展開（3次まで）
実行: cd 02_solutions/2016 && python3 verify.py
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30

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


a, b, c, d, lam = sp.symbols('a b c d lambda')
A = sp.Matrix([[a, 0, b, 0], [0, 1, 0, 0], [0, 0, 1, 0], [c, 0, 0, d]])

# === 〔I〕 行列式 =================================================
check("(I) |A| = ad", sp.expand(A.det()), sp.expand(a*d))

# === 〔II〕 逆行列 ================================================
Ainv = sp.simplify(A.inv())
print("A^{-1} ="); sp.pprint(Ainv)
check("(II) A*A^{-1}=I", sp.simplify(A*Ainv - sp.eye(4)) == sp.zeros(4), True)
Ainv_exp = sp.Matrix([[1/a, 0, -b/a, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [-c/(a*d), 0, b*c/(a*d), 1/d]])
check("(II) 逆行列が手計算と一致", sp.simplify(Ainv - Ainv_exp) == sp.zeros(4), True)

# === 〔III〕 固有値 ===============================================
cp = sp.expand(A.charpoly(lam).as_expr())
check("(III) char poly = (1-λ)^2 (a-λ)(d-λ)",
      cp, sp.expand((1-lam)**2*(a-lam)*(d-lam)))
note_unverifiable("(III) 固有値 1,1,a,d", "記号行列。特性多項式の因数分解で確認済み")

# === 〔IV〕 rank と nullity =======================================
F = sp.Matrix([[1, -3, 0], [0, 1, -1]])
check("(IV) rank = 2", F.rank(), 2)
check("(IV) dim ker = 1", F.shape[1] - F.rank(), 1)

# === 〔V〕 ∬ レムニスケート領域 ==================================
# 極座標: r^2 <= cos2θ, θ∈[-π/4,π/4]. 解析値 π/4 - 1/2
th, r = sp.symbols('theta r', real=True)
inner = sp.integrate(r/(1+r**2)**2, (r, 0, sp.sqrt(sp.cos(2*th))))
val5 = sp.integrate(inner, (th, -sp.pi/4, sp.pi/4))
check("(V) 解析値 = π/4 - 1/2", sp.simplify(val5), sp.pi/4 - sp.Rational(1, 2))
# 数値二重積分でも照合（極座標）
numf = lambda t: float(mp.quad(
        lambda rr: rr/(1+rr**2)**2, [0, mp.sqrt(mp.cos(2*t))]))
num5 = mp.quad(numf, [-mp.pi/4, mp.pi/4])
check("(V) 数値照合", float(num5), float(sp.pi/4 - sp.Rational(1, 2)), tol=1e-7)

# === 〔VI〕 球座標ヤコビアン ======================================
rr, tt, pp = sp.symbols('r theta phi', positive=True)
X = rr*sp.sin(tt)*sp.cos(pp); Y = rr*sp.sin(tt)*sp.sin(pp); Z = rr*sp.cos(tt)
J = sp.simplify(sp.Matrix([X, Y, Z]).jacobian([rr, tt, pp]).det())
print("J =", J)
check("(VI) J = r^2 sinθ", sp.simplify(J - rr**2*sp.sin(tt)), 0)

# === 〔VII〕 sinh(x+y) 展開 =======================================
xs, ys = sp.symbols('x y')
e7 = sp.sinh(xs + ys)
s = e7.series(xs, 0, 4).removeO().series(ys, 0, 4).removeO()
poly = sp.Poly(sp.expand(s), xs, ys)
trunc = sum(co*xs**m*ys**n for (m, n), co in poly.terms() if m + n <= 3)
expected7 = (xs + ys) + (xs + ys)**3/6
check("(VII) 3次まで", sp.expand(trunc), sp.expand(expected7))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
