#!/usr/bin/env python3
"""AY2014 材料力学 検算。全 PASS を確認してから tex を書く。"""
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


# ================= 〔I〕 はり =================
P, L, b, x, E, Iz, w, h = sp.symbols('P L b x E I_z w h', positive=True)

RA, RC = sp.symbols('R_A R_C')
# 鉛直つり合い: RA + RC - P = 0
# A まわりモーメント(反時計正): 2L*RC - L*P - P*b = 0
sol = sp.solve([RA + RC - P, 2*L*RC - L*P - P*b], [RA, RC])
RA_v = sol[RA]; RC_v = sol[RC]
check("(I-1) R_A", RA_v, P*(L-b)/(2*L))
check("(I-1) R_C", RC_v, P*(L+b)/(2*L))
check("(I-1) C まわりモーメント=0", sp.simplify(-2*L*RA_v + L*P - P*b), 0)

# (2) Q, M
Q_AB = RA_v
Q_BC = RA_v - P
Q_CD = RA_v - P + RC_v
check("(I-2) Q_AB", Q_AB, P*(L-b)/(2*L))
check("(I-2) Q_BC", Q_BC, -P*(L+b)/(2*L))
check("(I-2) Q_CD=0", sp.simplify(Q_CD), 0)

M_AB = RA_v*x
M_BC = RA_v*x - P*(x-L) + P*b          # B で +Pb の偶力ジャンプ
M_BC_simpl = P*(L+b)*(2*L-x)/(2*L)
check("(I-2) M_BC 整理形", sp.simplify(M_BC - M_BC_simpl), 0)
check("(I-2) M(0)=0", M_AB.subs(x, 0), 0)
check("(I-2) M(L^-)", M_AB.subs(x, L), P*(L-b)/2)
check("(I-2) M(L^+)", M_BC.subs(x, L), P*(L+b)/2)
check("(I-2) M(2L)=0", sp.simplify(M_BC.subs(x, 2*L)), 0)
check("(I-2) 偶力ジャンプ=Pb", sp.simplify(M_BC.subs(x, L) - M_AB.subs(x, L)), P*b)

# (3) 最大曲げ応力
Mmax = P*(L+b)/2
Zsec = w*h**2/6
sigma_max = Mmax/Zsec
check("(I-3) σ_max", sigma_max, 3*P*(L+b)/(w*h**2))
check("(I-3) 断面係数Z", (w*h**3/12)/(h/2), w*h**2/6)
note_unverifiable("(I-3) 発生位置", "x=L 断面の上下縁 y=±h/2（図・幾何からの記述）")

# (4) たわみ
C1, C2, C3, C4, C5, C6 = sp.symbols('C1 C2 C3 C4 C5 C6')
v1p = sp.integrate(M_AB, x) + C1
v1  = sp.integrate(v1p, x) + C2
v2p = sp.integrate(M_BC_simpl, x) + C3
v2  = sp.integrate(v2p, x) + C4
v3p = C5
v3  = C5*x + C6
eqs = [
    v1.subs(x, 0),
    v1.subs(x, L) - v2.subs(x, L),
    v1p.subs(x, L) - v2p.subs(x, L),
    v2.subs(x, 2*L),
    v2.subs(x, 2*L) - v3.subs(x, 2*L),
    v2p.subs(x, 2*L) - v3p,
]
csol = sp.solve(eqs, [C1, C2, C3, C4, C5, C6], dict=True)[0]
v1s = v1.subs(csol); v2s = v2.subs(csol); v3s = v3.subs(csol)
vB = sp.simplify(v1s.subs(x, L))
vD = sp.simplify(v3s.subs(x, 3*L))
print("  EI v_B =", vB)
print("  EI v_D =", vD)
check("(I-4) v(0)=0", v1s.subs(x, 0), 0)
check("(I-4) v(2L)=0", sp.simplify(v2s.subs(x, 2*L)), 0)
check("(I-4) 連続@L", sp.simplify(v1s.subs(x, L) - v2s.subs(x, L)), 0)
check("(I-4) EI*v_B 閉形式", vB, -P*L**3/6)
check("(I-4) EI*v_D 閉形式", vD, P*L**2*(3*L + b)/12)
# v_D は CD が剛体回転(M=0,Q=0): v_D = slopeC * L
slopeC = sp.simplify(v3s.diff(x))
check("(I-4) v_D = slope_C * L", sp.simplify(slopeC*L - vD), 0)

# ================= 〔II〕 段付丸棒 =================
A1, A2, Ip1, Ip2, L1, L2, G, alpha, dT, T0, d1, d2 = sp.symbols(
    'A_1 A_2 I_p1 I_p2 L_1 L_2 G alpha DeltaT T_0 d_1 d_2', positive=True)

# (1) 熱応力
N = sp.symbols('N')
Nsol = sp.solve(alpha*dT*(L1+L2) + N*L1/(E*A1) + N*L2/(E*A2), N)[0]
sig1 = sp.simplify(Nsol/A1)
sig2 = sp.simplify(Nsol/A2)
sig1_exp = -E*alpha*dT*(L1+L2)/(L1 + A1*L2/A2)
sig2_exp = -E*alpha*dT*(L1+L2)/(L2 + A2*L1/A1)
check("(II-1) σ1", sp.simplify(sig1 - sig1_exp), 0)
check("(II-1) σ2", sp.simplify(sig2 - sig2_exp), 0)
check("(II-1) σ1<0(圧縮)", float(sig1.subs({E:1,alpha:1,dT:1,L1:1,L2:1,A1:1,A2:2})) < 0, True)

# (2) ねじり
TA, TC = sp.symbols('T_A T_C')
tsol = sp.solve([TA+TC-T0, TA*L1/(G*Ip1) - TC*L2/(G*Ip2)], [TA, TC])
TA_v = sp.simplify(tsol[TA]); TC_v = sp.simplify(tsol[TC])
D = L1*Ip2 + L2*Ip1
check("(II-2) T_A", sp.simplify(TA_v - T0*L2*Ip1/D), 0)
check("(II-2) T_C", sp.simplify(TC_v - T0*L1*Ip2/D), 0)
check("(II-2) T_A+T_C=T0", sp.simplify(TA_v+TC_v), T0)
tauAB = sp.simplify(TA_v*(d1/2)/Ip1)
tauBC = sp.simplify(TC_v*(d2/2)/Ip2)
check("(II-2) tau_AB", sp.simplify(tauAB - T0*L2*d1/(2*D)), 0)
check("(II-2) tau_BC", sp.simplify(tauBC - T0*L1*d2/(2*D)), 0)
psiB = sp.simplify(TA_v*L1/(G*Ip1))
check("(II-2) psi_B", sp.simplify(psiB - T0*L1*L2/(G*D)), 0)
check("(II-2) psi_B 両側一致", sp.simplify(TA_v*L1/(G*Ip1) - TC_v*L2/(G*Ip2)), 0)

# (3) 点D モール円
sx, tau = sp.symbols('sigma_x tau')
center = sx/2
R = sp.sqrt((sx/2)**2 + tau**2)
smax = center + R
smin = center - R
tmax = R
nv = {sx: -100, tau: 30}
check("(II-3) σmax 数値", float(smax.subs(nv)), -50 + (50**2+30**2)**0.5)
check("(II-3) τmax 数値", float(tmax.subs(nv)), (50**2+30**2)**0.5)
check("(II-3) σmax+σmin=σx", sp.simplify(smax+smin - sx), 0)
note_unverifiable("(II-3) モール円作図/微小要素矢印", "作図・図示は数値検算対象外")


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
