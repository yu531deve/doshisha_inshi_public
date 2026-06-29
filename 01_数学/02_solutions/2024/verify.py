#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2024
=========================================
〔I〕  log(1+x) の Taylor 展開
〔II〕 ラプラシアンの極座標表示
〔III〕∫_{-∞}^{∞} e^{-(x-a)^2/b} dx
〔IV〕 ブロック対角行列の正則条件と逆行列
実行: cd 02_solutions/2024 && python3 verify.py
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


x = sp.symbols('x')

# === 〔I〕 log(1+x) 展開 =========================================
taylor = sp.series(sp.log(1+x), x, 0, 7).removeO()
expected1 = x - x**2/2 + x**3/3 - x**4/4 + x**5/5 - x**6/6
check("(I) Taylor (6次まで)", sp.expand(taylor), sp.expand(expected1))
k = sp.symbols('k', integer=True, positive=True)
partial = sum((-1)**(i+1)*x**i/i for i in range(1, 7))
check("(I) 一般項 Σ(-1)^{n+1}x^n/n", sp.expand(partial), sp.expand(expected1))

# === 〔II〕 ラプラシアンの極座標表示（具体関数で確認） ============
r, th = sp.symbols('r theta', positive=True)
xx, yy = sp.symbols('x y', real=True)
for expr_xy in [xx**2 + yy**2, xx*yy, sp.exp(xx)*sp.cos(yy), xx**3 - yy**2]:
    lap_xy = (sp.diff(expr_xy, xx, 2) + sp.diff(expr_xy, yy, 2)).subs(
        {xx: r*sp.cos(th), yy: r*sp.sin(th)})
    z = expr_xy.subs({xx: r*sp.cos(th), yy: r*sp.sin(th)})
    rhs = sp.diff(z, r, 2) + sp.diff(z, r)/r + sp.diff(z, th, 2)/r**2
    check(f"(II) Δ = z_rr+z_r/r+z_θθ/r^2 [{expr_xy}]", sp.simplify(lap_xy - rhs), 0)

# === 〔III〕 ガウス積分 ==========================================
a, b = sp.symbols('a b', real=True, positive=True)
val = sp.integrate(sp.exp(-(x-a)**2/b), (x, -sp.oo, sp.oo))
check("(III) ∫ = √(πb)", sp.simplify(val), sp.sqrt(sp.pi*b))

# === 〔IV〕 ブロック対角の正則条件と逆行列 =======================
aa, bb, cc, dd, p, q, rr2, ss = sp.symbols('a b c d p q r s', nonzero=True)
A = sp.Matrix([[aa, bb, 0, 0], [cc, dd, 0, 0], [0, 0, p, q], [0, 0, rr2, ss]])
check("(IV) det A = (ad-bc)(ps-qr)",
      sp.expand(A.det()), sp.expand((aa*dd-bb*cc)*(p*ss-q*rr2)))
# 逆行列（条件 ad-bc≠0, ps-qr≠0 のもと）
D1 = aa*dd - bb*cc
D2 = p*ss - q*rr2
Ainv = sp.Matrix([
    [dd/D1, -bb/D1, 0, 0],
    [-cc/D1, aa/D1, 0, 0],
    [0, 0, ss/D2, -q/D2],
    [0, 0, -rr2/D2, p/D2]])
check("(IV) A*A^{-1}=I", sp.simplify(A*Ainv - sp.eye(4)) == sp.zeros(4), True)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
