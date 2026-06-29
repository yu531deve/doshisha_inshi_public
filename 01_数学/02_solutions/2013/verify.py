#!/usr/bin/env python3
"""
解答検算 — 同志社 医工学 院試 数学 AY2013
=========================================
〔I〕  4x4 行列式（文字係数）
〔II〕 B の固有値・固有ベクトル
〔III〕Gram-Schmidt 正規直交化
〔IV〕 k∫e^{-x^2/2}=1 の k
〔V〕  3重積分（単体領域）
〔VI〕 合成関数の3階導関数
実行: cd 02_solutions/2013 && python3 verify.py
"""
import sympy as sp
mp_dps = 30

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


# === 〔I〕 4x4 行列式 ==============================================
a, b, c, d, e, f, g, h = sp.symbols('a b c d e f g h')
A = sp.Matrix([[a, b, c, d], [0, 1, 0, 0], [e, f, g, h], [0, 0, 1, 1]])
detA = sp.expand(A.det())
check("(I) |A| = ag-ah-ce+de", detA, sp.expand(a*g - a*h - c*e + d*e))

# === 〔II〕 固有値・固有ベクトル ===================================
B = sp.Matrix([[1, 0, 1], [-1, 2, 1], [0, 0, 2]])
eig = B.eigenvals()
print("B eigenvals =", eig)
check("(II) 固有値 {1,2(重複2)}",
      eig == {sp.Integer(1): 1, sp.Integer(2): 2}, True)
# λ=1 固有ベクトル (1,1,0)
v1 = sp.Matrix([1, 1, 0])
check("(II) λ=1: (B-I)v=0", (B - sp.eye(3))*v1 == sp.zeros(3, 1), True)
# λ=2 固有ベクトル (1,0,1),(0,1,0)
v2a = sp.Matrix([1, 0, 1])
v2b = sp.Matrix([0, 1, 0])
check("(II) λ=2: (B-2I)v2a=0", (B - 2*sp.eye(3))*v2a == sp.zeros(3, 1), True)
check("(II) λ=2: (B-2I)v2b=0", (B - 2*sp.eye(3))*v2b == sp.zeros(3, 1), True)
check("(II) λ=2 の幾何的重複度2", (B - 2*sp.eye(3)).rank(), 1)

# === 〔III〕 Gram-Schmidt =========================================
a1 = sp.Matrix([1, 0, 1]); a2 = sp.Matrix([1, 1, 0]); a3 = sp.Matrix([0, 1, 1])
e1 = sp.Matrix([1, 0, 1]) / sp.sqrt(2)
e2 = sp.Matrix([1, 2, -1]) / sp.sqrt(6)
e3 = sp.Matrix([-1, 1, 1]) / sp.sqrt(3)
# 正規性
for i, ev in enumerate([e1, e2, e3], 1):
    check(f"(III) |e{i}|=1", sp.simplify(ev.dot(ev)), 1)
# 直交性
check("(III) e1·e2=0", sp.simplify(e1.dot(e2)), 0)
check("(III) e1·e3=0", sp.simplify(e1.dot(e3)), 0)
check("(III) e2·e3=0", sp.simplify(e2.dot(e3)), 0)
# 張る空間が a1,a2,a3 と同じ（e1,e2 が a1,a2 の張る面内、e1 が a1 と平行）
check("(III) e1 ∥ a1", sp.simplify((e1.cross(a1)).norm()), 0)
# Gram-Schmidt を sympy 標準でも確認
gs = sp.GramSchmidt([a1, a2, a3], orthonormal=True)
check("(III) sympy GS と一致(符号許容)",
      all(sp.simplify((gs[i] - [e1, e2, e3][i]).norm()) == 0 or
          sp.simplify((gs[i] + [e1, e2, e3][i]).norm()) == 0 for i in range(3)), True)

# === 〔IV〕 規格化定数 ============================================
x = sp.symbols('x', real=True)
I4 = sp.integrate(sp.exp(-x**2/2), (x, -sp.oo, sp.oo))
k = 1/I4
check("(IV) k = 1/sqrt(2π)", sp.simplify(k), 1/sp.sqrt(2*sp.pi))

# === 〔V〕 3重積分（単体） =========================================
xx, yy, zz = sp.symbols('x y z', nonnegative=True)
a_ = sp.pi/2
V = sp.integrate(
        sp.integrate(
            sp.integrate(xx + yy + zz, (zz, 0, a_ - xx - yy)),
            (yy, 0, a_ - xx)),
        (xx, 0, a_))
check("(V) ∭(x+y+z) = π^4/128", sp.simplify(V), sp.pi**4/128)

# === 〔VI〕 合成関数の3階導関数 ===================================
t = sp.symbols('x')
ff = sp.Function('f'); gg = sp.Function('g')
z = gg(ff(t))
d3 = sp.diff(z, t, 3)
# 確定解答: g'''(f')^3 + 3 g''f'f'' + g' f'''
fp = sp.diff(ff(t), t); fpp = sp.diff(ff(t), t, 2); fppp = sp.diff(ff(t), t, 3)
gp = sp.Derivative(gg(ff(t)), ff(t))   # 記号的に扱いづらいので展開比較で代用
expected = (sp.diff(gg(ff(t)), ff(t), 3)*fp**3
            + 3*sp.diff(gg(ff(t)), ff(t), 2)*fp*fpp
            + sp.diff(gg(ff(t)), ff(t))*fppp)
check("(VI) d^3z/dx^3 公式", sp.simplify(d3 - expected), 0)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
