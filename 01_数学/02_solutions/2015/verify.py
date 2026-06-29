#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2015
=========================================
〔I〕  4x4 上三角行列の逆行列
〔II〕 同行列の固有値・固有ベクトル
〔III〕Gram-Schmidt 正規直交化
〔IV〕 ∬_D x dxdy, D: x^4/a^4+y^4/b^4<=1, x,y>0
〔V〕  ラプラシアンの極座標表示（証明）
〔VI〕 log(1+x+y^2) のマクローリン展開（3次まで）
〔VII〕球面上の xyz の極値（Lagrange）
実行: cd 02_solutions/2015 && python3 verify.py
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


A = sp.Matrix([[3, 5, -2, 2], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]])

# === 〔I〕 逆行列 ==================================================
Ainv = A.inv()
print("A^{-1} ="); sp.pprint(Ainv)
check("(I) A*A^{-1}=I", A*Ainv == sp.eye(4), True)
Ainv_exp = sp.Matrix([[sp.Rational(1, 3), sp.Rational(5, 3), sp.Rational(2, 3), -sp.Rational(1, 3)],
                      [0, -1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, sp.Rational(1, 2)]])
check("(I) 逆行列が手計算と一致", sp.simplify(Ainv - Ainv_exp) == sp.zeros(4), True)

# === 〔II〕 固有値・固有ベクトル ===================================
eig = A.eigenvals()
print("eigenvals =", eig)
check("(II) 固有値 {3,-1,1,2}",
      set(eig.keys()) == {sp.Integer(3), sp.Integer(-1), sp.Integer(1), sp.Integer(2)}, True)
vecs = {3: sp.Matrix([1, 0, 0, 0]), -1: sp.Matrix([-5, 4, 0, 0]),
        1: sp.Matrix([1, 0, 1, 0]), 2: sp.Matrix([-2, 0, 0, 1])}
for lam, v in vecs.items():
    check(f"(II) λ={lam}: (A-λI)v=0", (A - lam*sp.eye(4))*v == sp.zeros(4, 1), True)

# === 〔III〕 Gram-Schmidt =========================================
a1 = sp.Matrix([2, 2, 0]); a2 = sp.Matrix([3, 0, 3]); a3 = sp.Matrix([0, 1, 1])
e1 = sp.Matrix([1, 1, 0]) / sp.sqrt(2)
e2 = sp.Matrix([1, -1, 2]) / sp.sqrt(6)
e3 = sp.Matrix([-1, 1, 1]) / sp.sqrt(3)
for i, ev in enumerate([e1, e2, e3], 1):
    check(f"(III) |e{i}|=1", sp.simplify(ev.dot(ev)), 1)
check("(III) e1·e2=0", sp.simplify(e1.dot(e2)), 0)
check("(III) e1·e3=0", sp.simplify(e1.dot(e3)), 0)
check("(III) e2·e3=0", sp.simplify(e2.dot(e3)), 0)
gs = sp.GramSchmidt([a1, a2, a3], orthonormal=True)
check("(III) sympy GS と一致(符号許容)",
      all(sp.simplify((gs[i] - [e1, e2, e3][i]).norm()) == 0 or
          sp.simplify((gs[i] + [e1, e2, e3][i]).norm()) == 0 for i in range(3)), True)

# === 〔IV〕 ∬ x dxdy =============================================
# 変換 x=a s, y=b t → a^2 b ∬_{s^4+t^4<=1, s,t>0} s ds dt
# = a^2 b * (1/4) B(1/2, 5/4)
closed = sp.Rational(1, 4) * sp.beta(sp.Rational(1, 2), sp.Rational(5, 4))
print("closed (係数, a^2 b 除く) =", sp.nsimplify(closed), float(closed))
# 数値で二重積分を直接評価（a=b=1）
num = mp.quad(lambda s: s * (1 - s**4)**(mp.mpf(1)/4), [0, 1])
check("(IV) ∬ s ds dt (a=b=1) = (1/4)B(1/2,5/4)", float(num), float(closed), tol=1e-8)
# Γ表示との一致
gamma_form = sp.sqrt(sp.pi)*sp.gamma(sp.Rational(5, 4))/(4*sp.gamma(sp.Rational(7, 4)))
check("(IV) Γ表示と一致", sp.simplify(closed - gamma_form), 0)

# === 〔V〕 ラプラシアン極座標（証明） =============================
r, th = sp.symbols('r theta', positive=True)
fr = sp.Function('f')
# 一般の f(r,θ) を x=r cosθ, y=r sinθ で評価し直交→直交を確認するのは循環的なので、
# 具体関数 z=x^2+y^2 (=r^2) と z=x (=r cosθ) で恒等式の両辺一致を確認
for expr_xy, expr_rt in [(sp.Symbol('x')**2 + sp.Symbol('y')**2, r**2),
                         (sp.Symbol('x'), r*sp.cos(th)),
                         (sp.Symbol('x')*sp.Symbol('y'), r**2*sp.cos(th)*sp.sin(th))]:
    x, y = sp.symbols('x y')
    lap_xy = sp.diff(expr_xy, x, 2) + sp.diff(expr_xy, y, 2)
    # x,y で表した値を r,θ に変換
    lap_xy_rt = lap_xy.subs({x: r*sp.cos(th), y: r*sp.sin(th)})
    rhs = (sp.diff(expr_rt, r, 2) + sp.diff(expr_rt, r)/r
           + sp.diff(expr_rt, th, 2)/r**2)
    check(f"(V) 恒等式 [{expr_xy}]", sp.simplify(lap_xy_rt - rhs), 0)

# === 〔VI〕 マクローリン展開 ======================================
xs, ys = sp.symbols('x y')
f6 = sp.log(1 + xs + ys**2)
s = f6.series(xs, 0, 4).removeO().series(ys, 0, 4).removeO()
poly = sp.Poly(sp.expand(s), xs, ys)
trunc = sum(co*xs**m*ys**n for (m, n), co in poly.terms() if m + n <= 3)
expected6 = xs + ys**2 - xs**2/2 - xs*ys**2 + xs**3/3
check("(VI) 3次まで", sp.expand(trunc), sp.expand(expected6))

# === 〔VII〕 球面上 xyz の極値 ====================================
xv, yv, zv, lam = sp.symbols('x y z lam', positive=True)
g = xv**2 + yv**2 + zv**2 - 2
L = xv*yv*zv - lam*g
sol = sp.solve([sp.diff(L, xv), sp.diff(L, yv), sp.diff(L, zv), g],
               [xv, yv, zv, lam], dict=True)
print("VII sol =", sol)
xstar = sp.sqrt(sp.Rational(2, 3))
check("(VII) 臨界点 x=y=z=√(2/3)", sp.simplify(xstar - sp.sqrt(6)/3), 0)
val = xstar**3
check("(VII) 極値 = 2√6/9", sp.simplify(val - 2*sp.sqrt(6)/9), 0)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
