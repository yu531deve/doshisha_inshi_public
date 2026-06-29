#!/usr/bin/env python3
"""AY2013 材料力学 解答検算"""
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


# 記号
P, L, E, b, h, d, alpha, dT, x = sp.symbols('P L E b h d alpha dT x', positive=True)
I = b*h**3/12

# ============ 〔I〕 片持ちはり ============
# 荷重: B(x=L/2)に P下, C(x=L)に P下 と M0=5PL(時計回り)
M0 = 5*P*L

# (1) 反力. 鉛直: R_A=2P. モーメント(A,CCW正): M_A - P(L/2) - P L - 5PL = 0
R_A = 2*P
M_A = P*L/2 + P*L + 5*P*L
check("(1) R_A", R_A, 2*P)
check("(1) M_A (反力モーメント, CCW)", M_A, sp.Rational(13, 2)*P*L)

# (2) M(x) sagging正, 左側: M = sum F_up(x-a) - sum(CCW moment left)
M_AB = 2*P*x - sp.Rational(13, 2)*P*L          # 0<=x<L/2
M_BC = 2*P*x - sp.Rational(13, 2)*P*L - P*(x - L/2)  # L/2<=x<L
check("(2) M_AB simplified", M_AB, 2*P*x - sp.Rational(13, 2)*P*L)
check("(2) M_BC simplified", sp.expand(M_BC), sp.expand(P*x - 6*P*L))
check("(2) M(0)", M_AB.subs(x, 0), -sp.Rational(13, 2)*P*L)
check("(2) M(L/2) AB", M_AB.subs(x, L/2), -sp.Rational(11, 2)*P*L)
check("(2) M(L/2) BC 連続", M_BC.subs(x, L/2), -sp.Rational(11, 2)*P*L)
check("(2) M(L^-)", M_BC.subs(x, L), -5*P*L)
# 自由端Cで M0=5PL を加えて 0 になる
check("(2) M(L^-)+M0=0", M_BC.subs(x, L) + M0, 0)
Q_AB, Q_BC = 2*P, P
check("(2) Q_AB", Q_AB, 2*P)
check("(2) Q_BC", Q_BC, P)

# (3) B点上面モール円
M_B = sp.Rational(11, 2)*P*L      # |M_B|
sigma_x = M_B*(h/2)/I
check("(3) sigma_x at B top", sp.simplify(sigma_x), 33*P*L/(b*h**2))
tau_max = sigma_x/2
check("(3) tau_max", sp.simplify(tau_max), sp.Rational(33, 2)*P*L/(b*h**2))
note_unverifiable("(3) モール円作図", "図示部分(単軸応力状態の円)で数値検算対象外")

# (4) たわみ. EI v''=M, 固定端 v(0)=v'(0)=0
EIv1p = sp.integrate(M_AB, x)               # C1=0
EIv1 = sp.integrate(EIv1p, x)               # C2=0
C3, C4 = sp.symbols('C3 C4')
EIv2p = sp.integrate(M_BC, x) + C3
EIv2 = sp.integrate(EIv2p - C3, x) + C3*x + C4
sol = sp.solve([sp.Eq(EIv1p.subs(x, L/2), EIv2p.subs(x, L/2)),
                sp.Eq(EIv1.subs(x, L/2), EIv2.subs(x, L/2))], [C3, C4])
check("(4) C3", sol[C3], -P*L**2/8)
check("(4) C4", sol[C4], P*L**3/48)
EIv2p = EIv2p.subs(sol)
EIv2 = EIv2.subs(sol)
check("(4) EI theta_C", sp.simplify(EIv2p.subs(x, L)), -sp.Rational(45, 8)*P*L**2)
check("(4) EI v_C", sp.simplify(EIv2.subs(x, L)), -sp.Rational(47, 16)*P*L**3)
# 重ね合わせ検算
v_super = sp.Rational(5, 48)*P*L**3 + P*L**3/3 + 5*P*L*L**2/(2)  # M0 L^2/2 部分は /EI
check("(4) v_C 重ね合わせ", sp.simplify(v_super), sp.Rational(47, 16)*P*L**3)
th_super = P*(L/2)**2/2 + P*L**2/2 + 5*P*L*L  # theta, /EI
check("(4) theta_C 重ね合わせ", sp.simplify(th_super), sp.Rational(45, 8)*P*L**2)

# ============ 〔II〕 組合せ棒 ============
A1 = sp.pi/4*((8*d)**2 - (6*d)**2)
check("II A1", sp.simplify(A1), 7*sp.pi*d**2)
A2 = sp.pi/4*(d*(2 - x/L))**2
integ = sp.integrate(1/A2, (x, 0, L))
check("II ∫dx/A2", sp.simplify(integ), 2*L/(sp.pi*d**2))

# (1) 外力P. N1+N2=P, 適合: N1 L/(E A1)=∫N2/(2E A2)dx
N1, N2 = sp.symbols('N1 N2')
elong1 = N1*L/(E*A1)
elong2 = N2/(2*E)*integ
s1 = sp.solve([sp.Eq(N1+N2, P), sp.Eq(elong1, elong2)], [N1, N2])
check("(1) N1", s1[N1], sp.Rational(7, 8)*P)
check("(1) N2", s1[N2], sp.Rational(1, 8)*P)
sig1 = s1[N1]/A1
check("(1) sigma1", sp.simplify(sig1), P/(8*sp.pi*d**2))
sig2 = s1[N2]/A2
check("(1) sigma2(x)", sp.simplify(sig2), P/(2*sp.pi*d**2*(2 - x/L)**2))
check("(1) sigma2max (x=L)", sp.simplify(sig2.subs(x, L)), P/(2*sp.pi*d**2))
lam = elong1.subs(s1)
check("(1) lambda", sp.simplify(lam), P*L/(8*E*sp.pi*d**2))

# (2) 温度上昇 ΔT. 管2α, 棒α. NT1+NT2=0
NT1, NT2 = sp.symbols('NT1 NT2')
elT1 = 2*alpha*dT*L + NT1*L/(E*A1)
elT2 = alpha*dT*L + NT2/(2*E)*integ
s2 = sp.solve([sp.Eq(NT1+NT2, 0), sp.Eq(elT1, elT2)], [NT1, NT2])
check("(2) NT1 (圧縮)", sp.simplify(s2[NT1]), -sp.Rational(7, 8)*sp.pi*E*alpha*dT*d**2)
check("(2) NT2 (引張)", sp.simplify(s2[NT2]), sp.Rational(7, 8)*sp.pi*E*alpha*dT*d**2)
sigT1 = s2[NT1]/A1
check("(2) sigmaT1", sp.simplify(sigT1), -E*alpha*dT/8)
sigT2 = s2[NT2]/A2
check("(2) sigmaT2(x)", sp.simplify(sigT2), sp.Rational(7, 2)*E*alpha*dT/(2 - x/L)**2)
check("(2) sigmaT2max (x=L)", sp.simplify(sigT2.subs(x, L)), sp.Rational(7, 2)*E*alpha*dT)
lamT = elT1.subs(s2)
check("(2) lambdaT", sp.simplify(lamT), sp.Rational(15, 8)*alpha*dT*L)

note_unverifiable("II 自由体図", "図からのモデル化(並列・剛体板で伸び等しい)で数値検算対象外")

if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
