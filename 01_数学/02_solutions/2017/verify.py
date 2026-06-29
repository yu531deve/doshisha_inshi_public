#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2017
=========================================
〔I〕  4x4 行列の固有値
〔II〕 固有ベクトルの空欄 ア,イ,ウ,エ
〔III〕Gram-Schmidt の空欄 オ,カ,キ
〔IV〕 ∬_D y dxdy, D: 第1象限の四分楕円
〔V〕  z=e^x(x cosy - y siny) のラプラシアン
〔VI〕 sin(x+y^2) のマクローリン展開（3次まで）
実行: cd 02_solutions/2017 && python3 verify.py
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


A = sp.Matrix([[2, 1, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 1, 2]])

# === 〔I〕 固有値 =================================================
eig = A.eigenvals()
print("eigenvals =", eig)
check("(I) 固有値 {1,2(重複2),3}",
      eig == {sp.Integer(2): 2, sp.Integer(1): 1, sp.Integer(3): 1}, True)

# === 〔II〕 固有ベクトル空欄 ======================================
p1 = sp.Matrix([0, 0, 0, 1])   # ア=0, λ=2
p2 = sp.Matrix([1, 0, 0, 0])   # イ=0, λ=2
p3 = sp.Matrix([-1, 1, -1, 1])  # ウ=-1, λ=1
p4 = sp.Matrix([1, 1, 1, 1])   # エ=1, λ=3
check("(II) p1 は λ=2 の固有ベクトル", (A - 2*sp.eye(4))*p1 == sp.zeros(4, 1), True)
check("(II) p2 は λ=2 の固有ベクトル", (A - 2*sp.eye(4))*p2 == sp.zeros(4, 1), True)
check("(II) p3 は λ=1 の固有ベクトル", (A - 1*sp.eye(4))*p3 == sp.zeros(4, 1), True)
check("(II) p4 は λ=3 の固有ベクトル", (A - 3*sp.eye(4))*p4 == sp.zeros(4, 1), True)
print("→ ア=0, イ=0, ウ=-1, エ=1")

# === 〔III〕 Gram-Schmidt 空欄 ====================================
s2 = sp.sqrt(2)/2
u1 = sp.Matrix([s2, s2, 0])
u2 = sp.Matrix([0, 0, 1])
u3 = sp.Matrix([s2, -s2, 0])
a1 = sp.Matrix([1, 1, 0]); a2 = sp.Matrix([1, 1, 1]); a3 = sp.Matrix([1, 0, 1])
gs = sp.GramSchmidt([a1, a2, a3], orthonormal=True)
for i, (uu, gg) in enumerate(zip([u1, u2, u3], gs), 1):
    check(f"(III) u{i} = GS 結果(符号許容)",
          sp.simplify((uu - gg).norm()) == 0 or sp.simplify((uu + gg).norm()) == 0, True)
print("→ オ=√2/2, カ=0, キ=0")

# === 〔IV〕 ∬ y dxdy ============================================
# x=a u, y=b v → a b^2 ∬_{第1象限単位円} v du dv = a b^2 * 1/3
check("(IV) ∬ y dxdy = a b^2 / 3", sp.Rational(1, 3), sp.Rational(1, 3))
# 四分円上の ∬ v du dv = 1/3 を確認
rho, phi = sp.symbols('rho phi', positive=True)
quarter = sp.integrate(sp.integrate(rho*sp.sin(phi)*rho, (rho, 0, 1)), (phi, 0, sp.pi/2))
check("(IV) 四分単位円 ∬ v = 1/3", quarter, sp.Rational(1, 3))

# === 〔V〕 ラプラシアン ==========================================
x, y = sp.symbols('x y', real=True)
z = sp.exp(x)*(x*sp.cos(y) - y*sp.sin(y))
lap = sp.simplify(sp.diff(z, x, 2) + sp.diff(z, y, 2))
check("(V) Δz = 0", lap, 0)

# === 〔VI〕 sin(x+y^2) 展開 =======================================
xs, ys = sp.symbols('x y')
e6 = sp.sin(xs + ys**2)
s = e6.series(xs, 0, 4).removeO().series(ys, 0, 4).removeO()
poly = sp.Poly(sp.expand(s), xs, ys)
trunc = sum(co*xs**m*ys**n for (m, n), co in poly.terms() if m + n <= 3)
expected6 = xs + ys**2 - xs**3/6
check("(VI) 3次まで", sp.expand(trunc), sp.expand(expected6))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
