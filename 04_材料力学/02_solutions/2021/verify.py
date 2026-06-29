#!/usr/bin/env python3
"""AY2021 材料力学 検算 — 同志社 医工学 院試"""
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


P, d, L, E, G, T, F, a_q = sp.symbols('P d L E G T F a_q', positive=True)

# ===== 〔I〕 組合せ棒 =====
A_rod = sp.pi*d**2/4                       # 丸棒1,2 (直径d)
A_tube = sp.pi*((3*d)**2 - (2*d)**2)       # 円管3 (外半径3d,内半径2d)
check("(I) 円管断面積 A3", A_tube, 5*sp.pi*d**2)

# 節点変位 uT(上部剛体板), uB(点B), 上向き正, 底固定(u=0)
uT, uB = sp.symbols('uT uB')
N1 = (E*A_rod/(L/2))*(uT - uB)
N2 = (3*E*A_rod/(L/2))*(uB - 0)
N3 = (2*E*A_tube/L)*(uT - 0)
# 釣合い: 節点B  N1 - N2 - P = 0 ; 節点上板  -N1 - N3 = 0
sol = sp.solve([sp.Eq(N1 - N2 - P, 0), sp.Eq(-N1 - N3, 0)], [uT, uB], dict=True)[0]
N1v = sp.simplify(N1.subs(sol)); N2v = sp.simplify(N2.subs(sol)); N3v = sp.simplify(N3.subs(sol))
check("(I-1) N1", N1v, sp.Rational(20,83)*P)
check("(I-1) N2", N2v, -sp.Rational(63,83)*P)
check("(I-1) N3", N3v, -sp.Rational(20,83)*P)
check("(I-1) 釣合い N1-N2", sp.simplify(N1v-N2v), P)
check("(I-1) 釣合い N1+N3", sp.simplify(N1v+N3v), 0)

check("(I-2) σ1", sp.simplify(N1v/A_rod), 80*P/(83*sp.pi*d**2))
check("(I-2) σ2", sp.simplify(N2v/A_rod), -252*P/(83*sp.pi*d**2))
check("(I-2) σ3", sp.simplify(N3v/A_tube), -4*P/(83*sp.pi*d**2))

lam1 = N1v*(L/2)/(E*A_rod)
lam2 = N2v*(L/2)/(3*E*A_rod)
lam3 = N3v*L/(2*E*A_tube)
check("(I-3) λ1", sp.simplify(lam1), 40*P*L/(83*E*sp.pi*d**2))
check("(I-3) λ2", sp.simplify(lam2), -42*P*L/(83*E*sp.pi*d**2))
check("(I-3) λ3", sp.simplify(lam3), -2*P*L/(83*E*sp.pi*d**2))
check("(I-3) 適合 λ3=λ1+λ2", sp.simplify(lam3-(lam1+lam2)), 0)

J_rod = sp.pi*d**4/32
J_tube = sp.pi*((6*d)**4 - (4*d)**4)/32
check("(I-4) J3", J_tube, sp.Rational(1040,32)*sp.pi*d**4)
k_rod = G*J_rod/L          # 内棒直列: ねじれ = T_rod*L/(GJ)
k_tube = G*J_tube/L
Psi = T/(k_rod + k_tube)
check("(I-4) Ψ", sp.simplify(Psi), 32*T*L/(1041*sp.pi*G*d**4))

# ===== 〔II〕 円形断面はり + 剛体棒 =====
I_sec = sp.pi*d**4/64
J_sec = sp.pi*d**4/32
R_A = F; M_A = F*L; T_A = F*L
check("(II-1) 反力", R_A, F)
check("(II-1) 支持モーメント", M_A, F*L)
check("(II-1) 支持トルク", T_A, F*L)
x = sp.symbols('x', nonnegative=True)
Mx = F*(L-x)
check("(II-2) M(0)=FL", Mx.subs(x,0), F*L)
check("(II-2) M(L)=0", Mx.subs(x,L), 0)
M_Q = Mx.subs(x, L-a_q)
check("(II-3) M_Q", sp.simplify(M_Q), F*a_q)
sx = M_Q*(d/2)/I_sec
check("(II-3) σx", sp.simplify(sx), 32*F*a_q/(sp.pi*d**3))
check("(II-3) σy(円周方向=0)", 0, 0)
tau = (F*L)*(d/2)/J_sec
check("(II-3) τxy", sp.simplify(tau), 16*F*L/(sp.pi*d**3))
s = sp.simplify(sx); t = sp.simplify(tau)
smax = s/2 + sp.sqrt((s/2)**2 + t**2)
smin = s/2 - sp.sqrt((s/2)**2 + t**2)
check("(II-4) σmax", sp.simplify(smax),
      sp.simplify(16*F/(sp.pi*d**3)*(a_q + sp.sqrt(a_q**2 + L**2))))
check("(II-4) σmin", sp.simplify(smin),
      sp.simplify(16*F/(sp.pi*d**3)*(a_q - sp.sqrt(a_q**2 + L**2))))
tau_max = sp.sqrt((s/2)**2 + t**2)
check("(II-4) τmax", sp.simplify(tau_max),
      sp.simplify(16*F/(sp.pi*d**3)*sp.sqrt(a_q**2 + L**2)))

note_unverifiable("(I) 上部剛体板の境界条件", "板は床に固定されず軸方向に平行移動可(図からのモデル化)")
note_unverifiable("(II) 剛体棒の向き", "原本3次元図より梁軸直交方向と判断(支持トルク発生のため)")

if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
