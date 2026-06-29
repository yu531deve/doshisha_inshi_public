#!/usr/bin/env python3
"""
AY2012 材料力学 解答検算
〔I〕単純支持・突き出しはり（右端C点に集中荷重P）
〔II〕上端固定の台形板の引張り
符号規約: 反力・たわみは上向きを正。曲げモーメント M は下に凸（さがり）を正。
         はりの基礎式は EI v'' = M(x) を用いる。
実行: cd 02_solutions/2012 && python3 verify.py
"""
import sympy as sp

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


P, L, E, I, x = sp.symbols('P L E I x', positive=True)
b1, b2, t = sp.symbols('b1 b2 t', positive=True)

# ===== 〔I〕単純支持・突き出しはり =====
# 支点 A(x=0, ピン), B(x=L, ローラー), 右端 C(x=2L) に集中荷重 P(下向き)
RA = sp.symbols('RA'); RB = sp.symbols('RB')
sol = sp.solve([RA + RB - P,            # ΣF_y=0 (上向き正, 外力 P は下向き)
                RB*L - P*(2*L)], [RA, RB])  # ΣM_A=0
check("(I-1) R_A", sol[RA], -P)
check("(I-1) R_B", sol[RB], 2*P)

RA_v, RB_v = -P, 2*P
# 区間I 0<=x<L: M1 = RA x ;  区間II L<=x<=2L: M2 = RA x + RB (x-L)
M1 = RA_v*x
M2 = RA_v*x + RB_v*(x - L)
check("(I-2) M(L^-)=M(L^+) 連続", M1.subs(x, L), M2.subs(x, L))
check("(I-2) M(2L)=0 (自由端)", sp.simplify(M2.subs(x, 2*L)), 0)
check("(I-2) M(L) = -PL (B点)", M1.subs(x, L), -P*L)

# (3) たわみ: EI v'' = M
C1, C2, C3, C4 = sp.symbols('C1 C2 C3 C4')
v1 = sp.integrate(sp.integrate(M1, x), x) + C1*x + C2   # ×EI
v2 = sp.integrate(sp.integrate(M2, x), x) + C3*x + C4   # ×EI
bc = sp.solve([v1.subs(x, 0),                 # v1(0)=0
               v1.subs(x, L),                 # v1(L)=0
               (v1 - v2).subs(x, L),          # 連続 v
               (sp.diff(v1, x) - sp.diff(v2, x)).subs(x, L)],  # 連続 v'
              [C1, C2, C3, C4])
v2c = v2.subs(bc)
deltaC = sp.simplify(v2c.subs(x, 2*L))        # = EI*v_C
check("(I-3) EI*v_C = -2PL^3/3 (下向き)", deltaC, -sp.Rational(2, 3)*P*L**3)

# (4) カスチリアーノ: δ_C = ∂U/∂P = ∫ M (∂M/∂P)/EI dx
U_int = sp.integrate(M1*sp.diff(M1, P), (x, 0, L)) + sp.integrate(M2*sp.diff(M2, P), (x, L, 2*L))
deltaC_energy = sp.simplify(U_int)            # = EI*δ_C (Pに対する正方向=下向き)
check("(I-4) カスチリアーノ δ_C = 2PL^3/3EI", deltaC_energy, sp.Rational(2, 3)*P*L**3)

# ===== 〔II〕台形板の引張り =====
# 幅 b(x)=b1+(b2-b1)x/L (x: 下端B=0 → 上端A=L), 断面積 A(x)=t b(x)
bx = b1 + (b2 - b1)*x/L
Ax = t*bx
check("(II-1) A(0)=t b1", Ax.subs(x, 0), t*b1)
check("(II-1) A(L)=t b2", Ax.subs(x, L), t*b2)
# (2) N=P 一定, σ=P/A
sigma = P/Ax
# (3) 伸び λ=∫σ/E dx
lam = sp.simplify(sp.integrate(sigma/E, (x, 0, L)))
lam_expected = P*L*sp.log(b2/b1)/(E*t*(b2 - b1))
check("(II-3) 伸び λ", sp.simplify(lam - lam_expected), 0)
# 一様幅 b2->b1 の極限で λ -> PL/(E t b1)
lam_uniform = sp.limit(lam_expected, b2, b1)
check("(II-3) 極限 b2->b1 で λ=PL/(E t b1)", sp.simplify(lam_uniform), P*L/(E*t*b1))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
