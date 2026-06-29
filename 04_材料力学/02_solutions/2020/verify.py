#!/usr/bin/env python3
"""AY2020 材料力学 検算 — 全小問 sympy。"""
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


# ===== 〔I〕 両端固定の段付き棒（静定不能）=====
E, A, L, P1, P2, Ra = sp.symbols('E A L P1 P2 Ra', positive=True)
# 符号: x 右向き正, 軸力 N 引張正, 変位 右向き正.
# 節点 a=0, b=L/2, c=L, d=L+L/3, e=2L.
# P1 は b 点で左向き(-x), P2 は d 点で右向き(+x).
# 内力 N = -(左側にある外力の +x 成分の和)
N_ab = -Ra                  # 区間 a-b
N_bc = -(Ra - P1)           # 区間 b-c  (左に -P1)
N_cd = -(Ra - P1)           # 区間 c-d  (c で外力なし)
N_de = -(Ra - P1 + P2)      # 区間 d-e  (左に +P2)

# 全体つり合いの確認
Re = sp.symbols('Re')
sol_eq = sp.solve(sp.Eq(Ra + Re - P1 + P2, 0), Re)[0]
check("(I) 全体つり合い Re", sol_eq, P1 - P2 - Ra)

# 各棒の伸び  EA1 = 3E*A, EA2 = E*2A
EA1 = 3 * E * A
EA2 = E * 2 * A
lam1 = (N_ab * (L / 2) + N_bc * (L / 2)) / EA1
lam2 = (N_cd * (L / 3) + N_de * (2 * L / 3)) / EA2

# 適合条件: 両端固定 → 全伸び 0
Ra_sol = sp.solve(sp.Eq(lam1 + lam2, 0), Ra)[0]
check("(I) 反力 Ra", Ra_sol, (4 * P1 - 2 * P2) / 5)

lam1_f = sp.simplify(lam1.subs(Ra, Ra_sol))
lam2_f = sp.simplify(lam2.subs(Ra, Ra_sol))
check("(I)(1) λ1", lam1_f, L * (4 * P2 - 3 * P1) / (30 * E * A))
check("(I)(1) λ2", lam2_f, L * (3 * P1 - 4 * P2) / (30 * E * A))
check("(I)(1) λ1+λ2=0", sp.simplify(lam1_f + lam2_f), 0)

# (2) c の変位 = a からの全伸び = λ1 = 0
P2_sol = sp.solve(sp.Eq(lam1_f, 0), P2)[0]
check("(I)(2) P2 (c の変位0)", P2_sol, 3 * P1 / 4)
check("(I)(2) λ1=0 確認", lam1_f.subs(P2, sp.Rational(3, 4) * P1), 0)

# ===== 〔II〕 片持ちはり =====
x, P, M0, L2 = sp.symbols('x P M0 L', positive=True)
# A 固定(x=0), B: x=L で時計回り M0, C: x=2L で下向き P.
# 符号: たわみ上向き正, 曲げM さがり(凹上)正, EI v'' = M.
M2 = -P * (2 * L2 - x)               # L<=x<=2L
M1 = -P * (2 * L2 - x) - M0          # 0<=x<L (右側に M0 時計回り=-M0)

check("(II)(1) M(0) 固定端", M1.subs(x, 0), -2 * P * L2 - M0)
check("(II)(1) M(2L) 自由端", M2.subs(x, 2 * L2), 0)
check("(II)(1) B でのBMDジャンプ", sp.simplify(M2.subs(x, L2) - M1.subs(x, L2)), M0)
check("(II)(1) Q=dM/dx 一定 region1", sp.diff(M1, x), P)
check("(II)(1) Q region2", sp.diff(M2, x), P)

# (2) たわみ: EI v'' = M1, v(0)=0, v'(0)=0
C1, C2 = sp.symbols('C1 C2')
EIv1p = sp.integrate(M1, x) + C1
EIv1 = sp.integrate(EIv1p, x) + C2
sol = sp.solve([EIv1p.subs(x, 0), EIv1.subs(x, 0)], [C1, C2])
EIv1 = EIv1.subs(sol)
deltaB_EI = sp.simplify(EIv1.subs(x, L2))
check("(II)(2) EI*δB", deltaB_EI, -5 * P * L2**3 / 6 - M0 * L2**2 / 2)
check("(II)(2) δB 大きさ", -deltaB_EI, 5 * P * L2**3 / 6 + M0 * L2**2 / 2)
check("(II)(2) P単独照合", (-deltaB_EI).subs(M0, 0), 5 * P * L2**3 / 6)

note_unverifiable("(II)(1) SFD/BMD 作図", "図示は描画であり数値検算の対象外（値は上記で検算済）")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
