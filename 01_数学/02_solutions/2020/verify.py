#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2020
=========================================
〔I〕  勾配ノルムの極座標表示（証明）
〔II〕 ∬_D 1/√(a^2-x^2-y^2) dxdy（円板）
〔III〕A の対角化可能性
〔IV〕 スカラー三重積 (a×b,c)=(a,b×c) と四面体の体積
実行: cd 02_solutions/2020 && python3 verify.py
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


# === 〔I〕 勾配ノルムの恒等式（具体関数で確認） ==================
r, th = sp.symbols('r theta', positive=True)
x, y = sp.symbols('x y', real=True)
for expr_xy in [x**2 + y**2, x*y, sp.exp(x)*sp.cos(y), x**3 - y]:
    zx = sp.diff(expr_xy, x); zy = sp.diff(expr_xy, y)
    lhs = (zx**2 + zy**2).subs({x: r*sp.cos(th), y: r*sp.sin(th)})
    z_rt = expr_xy.subs({x: r*sp.cos(th), y: r*sp.sin(th)})
    rhs = sp.diff(z_rt, r)**2 + sp.diff(z_rt, th)**2/r**2
    check(f"(I) 恒等式 [{expr_xy}]", sp.simplify(lhs - rhs), 0)

# === 〔II〕 ∬ 1/√(a^2-r^2) =====================================
a = sp.symbols('a', positive=True)
val = sp.integrate(sp.integrate(r/sp.sqrt(a**2 - r**2), (r, 0, a)), (th, 0, 2*sp.pi))
check("(II) ∬ = 2πa", sp.simplify(val), 2*sp.pi*a)

# === 〔III〕 対角化可能性 ========================================
A = sp.Matrix([[1, 1, 0], [0, 1, 0], [1, 0, 1]])
eig = A.eigenvals()
print("eigenvals =", eig)
check("(III) 固有値 λ=1 (重複度3)", eig == {sp.Integer(1): 3}, True)
geo = 3 - (A - sp.eye(3)).rank()
check("(III) 幾何的重複度 = 1 (<3) → 対角化不可", geo, 1)
check("(III) 対角化不可(固有ベクトル<3)", A.is_diagonalizable(), False)

# === 〔IV〕 スカラー三重積と体積 ================================
av = sp.Matrix(sp.symbols('a1 a2 a3'))
bv = sp.Matrix(sp.symbols('b1 b2 b3'))
cv = sp.Matrix(sp.symbols('c1 c2 c3'))
lhs = (av.cross(bv)).dot(cv)
rhs = av.dot(bv.cross(cv))
check("(IV) (a×b,c) = (a,b×c)", sp.expand(lhs - rhs), 0)
# = det[a b c]
detM = sp.Matrix.hstack(av, bv, cv).det()
check("(IV) (a×b,c) = det[a b c]", sp.expand(lhs - detM), 0)
# 体積 = (1/6)|det|: 単位ベクトルの四面体で 1/6
check("(IV) 標準四面体の体積 = 1/6",
      sp.Rational(1, 6)*abs(sp.Matrix.hstack(
          sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])).det()),
      sp.Rational(1, 6))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
