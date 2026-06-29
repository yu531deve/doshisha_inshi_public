#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2021
=========================================
〔I〕  陰関数 x-y+log(xy)=0 の d^2y/dx^2
〔II〕 ∭_D cos(x+y+z) dV, D: 単体
〔III〕4x4 行列の行列式・固有値・固有ベクトル
〔IV〕 部分空間 W1 の基底・次元, dim(W1+W2)
実行: cd 02_solutions/2021 && python3 verify.py
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


# === 〔I〕 陰関数の2階微分 =======================================
x = sp.symbols('x')
y = sp.Function('y')
F = x - y(x) + sp.log(x*y(x))
yp = sp.solve(sp.diff(F, x), sp.Derivative(y(x), x))[0]
print("y' =", sp.simplify(yp))
# y'' を求める
ypp = sp.diff(yp, x).subs(sp.Derivative(y(x), x), yp)
ypp = sp.simplify(ypp)
print("y'' =", ypp)
# 確定解答候補: y'(=(x+1)y/(x(y-1)))、y'' を数値で照合
# 数値照合: 具体的な (x0,y0) を曲線上で取り、有限差分と比較
yy = sp.symbols('y')
ypexpr = (x + 1)*yy/(x*(yy - 1))
check("(I) y' = (x+1)y/(x(y-1))",
      sp.simplify(yp.subs(y(x), yy) - ypexpr), 0)
ypp_form = -yy*((x+1)**2 + (yy-1)**2)/(x**2*(yy-1)**3)
check("(I) y'' = -y((x+1)^2+(y-1)^2)/(x^2(y-1)^3)",
      sp.simplify(ypp.subs(y(x), yy) - ypp_form), 0)
# 曲線 x - y + log(xy)=0 上の点で数値検証（x0=1 → 1 - y + log(y)=0 → y=1）
# y=1 は特異(y'分母0)なので別点。x0, y0 を数値解で求める
import mpmath as mp
mp.mp.dps = 25
x0 = mp.mpf('2.0')
y0 = mp.findroot(lambda yv: x0 - yv + mp.log(x0*yv), mp.mpf('4.5'))
# 陰関数を y(x) として有限差分で y'' を数値計算
def yfun(xv):
    return mp.findroot(lambda yv: xv - yv + mp.log(xv*yv), y0)
h = mp.mpf('1e-6')
ypp_num = (yfun(x0+h) - 2*yfun(x0) + yfun(x0-h))/h**2
ypp_sym = sp.lambdify((x, yy), ypp.subs(y(x), yy), 'mpmath')(float(x0), float(y0))
check("(I) y'' 数値照合", float(ypp_num), float(ypp_sym), tol=1e-4)

# === 〔II〕 ∭ cos(x+y+z) 単体 ===================================
xx, yy2, zz = sp.symbols('x y z', nonnegative=True)
a = sp.pi/2
V = sp.integrate(
        sp.integrate(
            sp.integrate(sp.cos(xx+yy2+zz), (zz, 0, a-xx-yy2)),
            (yy2, 0, a-xx)),
        (xx, 0, a))
check("(II) ∭ cos = π^2/8 - 1", sp.simplify(V), sp.pi**2/8 - 1)

# === 〔III〕 行列式・固有値・固有ベクトル ========================
a_, b_, c_, lam = sp.symbols('a b c lambda')
A = sp.Matrix([[a_, 1, 0, 0], [0, 1, 0, 0], [b_, 0, b_, 0], [0, 0, 1, c_]])
check("(III) |A| = abc", sp.expand(A.det()), sp.expand(a_*b_*c_))
cp = sp.expand(A.charpoly(lam).as_expr())
check("(III) char poly = (a-λ)(1-λ)(b-λ)(c-λ)",
      cp, sp.expand((a_-lam)*(1-lam)*(b_-lam)*(c_-lam)))
# 固有ベクトル（手計算の整形形）を検算
v_c = sp.Matrix([0, 0, 0, 1])
v_b = sp.Matrix([0, 0, b_-c_, 1])
v_a = sp.Matrix([(a_-b_)*(c_-a_), 0, b_*(c_-a_), -b_])
v_1 = sp.Matrix([(b_-1)*(c_-1), -(a_-1)*(b_-1)*(c_-1), -b_*(c_-1), b_])
check("(III) λ=c の固有ベクトル", (A - c_*sp.eye(4))*v_c == sp.zeros(4, 1), True)
check("(III) λ=b の固有ベクトル", sp.simplify((A - b_*sp.eye(4))*v_b) == sp.zeros(4, 1), True)
check("(III) λ=a の固有ベクトル", sp.simplify((A - a_*sp.eye(4))*v_a) == sp.zeros(4, 1), True)
check("(III) λ=1 の固有ベクトル", sp.simplify((A - 1*sp.eye(4))*v_1) == sp.zeros(4, 1), True)

# === 〔IV〕 部分空間 =============================================
# W1: x-2y+w=0, y+z+w=0  → dim 2
M1 = sp.Matrix([[1, -2, 0, 1], [0, 1, 1, 1]])
check("(IV) dim W1 = 2", 4 - M1.rank(), 2)
# 基底候補
basis = [sp.Matrix([2, 1, -1, 0]), sp.Matrix([-1, 0, -1, 1])]
for bvec in basis:
    check("(IV) 基底ベクトルが W1 を満たす", M1*bvec == sp.zeros(2, 1), True)
check("(IV) 基底は1次独立",
      sp.Matrix.hstack(*basis).rank(), 2)
# W2: x-y-z=0 → dim 3, W1∩W2 と W1+W2
M2 = sp.Matrix([[1, -1, -1, 0]])
check("(IV) dim W2 = 3", 4 - M2.rank(), 3)
Mcap = sp.Matrix([[1, -2, 0, 1], [0, 1, 1, 1], [1, -1, -1, 0]])
dim_cap = 4 - Mcap.rank()
check("(IV) dim(W1∩W2) = 1", dim_cap, 1)
check("(IV) dim(W1+W2) = 4", 2 + 3 - dim_cap, 4)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
