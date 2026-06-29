#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2014
=========================================
〔I〕  4x4 逆行列（文字係数）
〔II〕 4x4 行列式 = (a-x)(b-x)(c-x) の証明
〔III〕固有値・固有ベクトル
〔IV〕 ∫_0^1 x^{-a} dx の場合分け
〔V〕  ヤコビアン
〔VI〕 上半楕円体上の ∭ z dV
〔VII〕f=x^3-3xy+y^3 の極値
〔VIII〕e^{x+y^2} のマクローリン展開（3次まで）
実行: cd 02_solutions/2014 && python3 verify.py
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


a, b, c, f, g, h, x = sp.symbols('a b c f g h x')

# === 〔I〕 逆行列 ==================================================
A = sp.Matrix([[a, b, c, 0], [0, 1, 0, 0], [0, f, g, h], [0, 0, 1, 0]])
print("det A =", sp.simplify(A.det()))
Ainv = sp.simplify(A.inv())
print("A^{-1} =")
sp.pprint(Ainv)
check("(I) A*A^{-1}=I", sp.simplify(A*Ainv - sp.eye(4)) == sp.zeros(4), True)

# === 〔II〕 行列式の証明 ===========================================
aa, bb, cc, xx = sp.symbols('a b c x')
M = sp.Matrix([[xx, aa, 1, bb],
               [aa, xx, 1, bb],
               [aa, bb, 1, xx],
               [aa, bb, 1, cc]])
check("(II) det = (a-x)(b-x)(c-x)",
      sp.expand(M.det()), sp.expand((aa-xx)*(bb-xx)*(cc-xx)))

# === 〔III〕 固有値・固有ベクトル ==================================
B = sp.Matrix([[1, 0, 1], [-2, 2, 2], [0, 2, 4]])
eig = B.eigenvals()
print("B eigenvals =", eig)
check("(III) 固有値 {0,2,5}",
      set(eig.keys()) == {sp.Integer(0), sp.Integer(2), sp.Integer(5)}, True)
vecs = {0: sp.Matrix([-1, -2, 1]), 2: sp.Matrix([1, -1, 1]), 5: sp.Matrix([1, 2, 4])}
for lam, v in vecs.items():
    check(f"(III) λ={lam}: (B-λI)v=0", (B - lam*sp.eye(3))*v == sp.zeros(3, 1), True)

# === 〔IV〕 広義積分 ==============================================
ar = sp.symbols('a', positive=False, real=True)
# a<1 のとき収束: 1/(1-a)
val = sp.integrate(x**(-sp.Rational(1, 2)), (x, 0, 1))  # a=1/2 <1 の具体例
check("(IV) a=1/2 で 1/(1-a)=2", val, 2)
# a=2(>1)で発散
val_div = sp.integrate(x**(-2), (x, 0, 1))
check("(IV) a=2 で発散(+oo)", val_div, sp.oo)
note_unverifiable("(IV) 一般形 1/(1-a) (a<1)",
                  "場合分けの記号結果。a<1 の具体値で収束値を確認済み")

# === 〔V〕 ヤコビアン ==============================================
r, th, ph = sp.symbols('r theta phi', positive=True)
X = a*r*sp.sin(th)*sp.cos(ph)
Y = b*r*sp.sin(th)*sp.sin(ph)
Z = c*r*sp.cos(th)
Jac = sp.Matrix([X, Y, Z]).jacobian([r, th, ph])
J = sp.simplify(Jac.det())
print("J =", J)
check("(V) J = a b c r^2 sinθ", sp.simplify(J - a*b*c*r**2*sp.sin(th)), 0)

# === 〔VI〕 上半楕円体 ∭ z dV =====================================
# 変数変換 x=a u,... → a b c^2 ∭_{上半単位球} w dudvdw = a b c^2 * π/4
u, v, w = sp.symbols('u v w')
# 上半単位球で w を積分（球座標）
rho, ang = sp.symbols('rho ang', positive=True)
inner = sp.integrate(
            sp.integrate(
                sp.integrate(rho*sp.cos(ang) * rho**2*sp.sin(ang),
                             (rho, 0, 1)),
                (ang, 0, sp.pi/2)),
            (ph, 0, 2*sp.pi))
check("(VI) 上半単位球 ∭ w = π/4", sp.simplify(inner), sp.pi/4)
check("(VI) 全体 = π a b c^2 /4",
      sp.simplify(a*b*c**2 * inner), sp.pi*a*b*c**2/4)

# === 〔VII〕 極値 =================================================
xx2, yy2 = sp.symbols('x y', real=True)
F = xx2**3 - 3*xx2*yy2 + yy2**3
fx = sp.diff(F, xx2); fy = sp.diff(F, yy2)
crit = sp.solve([fx, fy], [xx2, yy2], dict=True)
print("critical points =", crit)
H = sp.hessian(F, (xx2, yy2))
# (1,1): 極小、値 -1
D11 = H.subs({xx2: 1, yy2: 1}).det()
check("(VII) (1,1) で D>0", D11 > 0, True)
check("(VII) (1,1) で f_xx>0", H[0, 0].subs({xx2: 1, yy2: 1}) > 0, True)
check("(VII) f(1,1) = -1", F.subs({xx2: 1, yy2: 1}), -1)
D00 = H.subs({xx2: 0, yy2: 0}).det()
check("(VII) (0,0) は鞍点 D<0", D00 < 0, True)

# === 〔VIII〕 マクローリン展開 ====================================
xs, ys = sp.symbols('x y')
expr = sp.exp(xs + ys**2)
# 全次数3まで
ser = expr.series(xs, 0, 4).removeO()
ser = ser.series(ys, 0, 4).removeO()
# 全次数<=3 の項のみ抽出
poly = sp.Poly(sp.expand(ser), xs, ys)
trunc = sum(coef*xs**m*ys**n for (m, n), coef in poly.terms() if m + n <= 3)
expected_v = 1 + xs + xs**2/2 + xs**3/6 + ys**2 + xs*ys**2
check("(VIII) 3次までの展開", sp.expand(trunc), sp.expand(expected_v))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
