#!/usr/bin/env python3
"""AY2025 材料力学 解答検算"""
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


# ===== 共通記号 =====
P, L, E, A, d, delta = sp.symbols('P L E A d delta', positive=True)
I, Ip, G = sp.symbols('I I_p G', positive=True)

# =========================================================
# 〔I〕剛体板 + 2本の棒① + 中央の棒②
#   棒①: 長さ2L, 断面積A, 縦弾性係数E, 2本（各々 P/2 を負担）
#   棒②: 長さL,  断面積2A, 縦弾性係数2E, 荷重P
# =========================================================
# (1) 幾何関係: 上面の下方移動 δ = λ1(板が下がる) + λ2(棒②が縮む)
lam1 = (P/2) * (2*L) / (A*E)        # 棒①1本の伸び（張力 P/2）
lam2 = P * L / ((2*A) * (2*E))      # 棒②の縮み（圧縮力 P）
check("(1) λ1 = PL/(AE)", lam1, P*L/(A*E))
check("(1) λ2 = PL/(4AE)", lam2, P*L/(4*A*E))

# (2) 荷重 P を δ で表す: δ = λ1 + λ2
P_sol = sp.solve(sp.Eq(delta, lam1 + lam2), P)[0]
check("(2) 荷重 P", P_sol, sp.Rational(4, 5)*A*E*delta/L)

# (3) 応力 σ1=(P/2)/A, σ2=P/(2A)
sigma1 = (P/2)/A
sigma2 = P/(2*A)
check("(3) σ1 (記号)", sigma1, P/(2*A))
check("(3) σ2 (記号)", sigma2, P/(2*A))
check("(3) σ1=σ2", sigma1, sigma2)
check("(3) σ1 = 2Eδ/(5L)", sigma1.subs(P, P_sol), 2*E*delta/(5*L))
check("(3) σ2 = 2Eδ/(5L)", sigma2.subs(P, P_sol), 2*E*delta/(5*L))

# (4) λ1, λ2
check("(4) λ1 = 4δ/5", lam1.subs(P, P_sol), sp.Rational(4, 5)*delta)
check("(4) λ2 = δ/5", lam2.subs(P, P_sol), sp.Rational(1, 5)*delta)
check("(4) λ1+λ2 = δ", (lam1+lam2).subs(P, P_sol), delta)

# =========================================================
# 〔II〕L字はり（直径d, A固定, AB=L(x), BC=L/2(y), C に下向きP）
#   A(0,0,0)-B(L,0,0)-C(L, L/2, 0), 荷重 F=(0,0,-P)
# =========================================================
rC = sp.Matrix([L, sp.Rational(1, 2)*L, 0])
F = sp.Matrix([0, 0, -P])
M_A = rC.cross(F)   # A まわりのモーメント
check("(1) M_A の x成分 (= -ねじり)", M_A[0], -P*L/2)
check("(1) M_A の y成分 (= 曲げ)", M_A[1], P*L)
check("(1) M_A の z成分", M_A[2], 0)
check("(1) 曲げモーメント大きさ", sp.Abs(M_A[1]), P*L)
check("(1) ねじりトルク大きさ", sp.Abs(M_A[0]), P*L/2)

# (2) 区間AB: 断面x で C側自由体
xx = sp.symbols('xx', positive=True)
Q_AB = P
M_AB = P*(L - xx)
T_AB = P*(L/2)
check("(2) Q(x)=P", Q_AB, P)
check("(2) M(0)=PL", M_AB.subs(xx, 0), P*L)
check("(2) M(L)=0", M_AB.subs(xx, L), 0)
check("(2) T=PL/2", T_AB, P*L/2)

# (3) 断面D (x=L/4), 点u (z=+d/2 最上点)
M_D = M_AB.subs(xx, L/4)
T_D = T_AB
check("(3) M_D = 3PL/4", M_D, sp.Rational(3, 4)*P*L)
sigma_x = M_D*(d/2)/I
sigma_y = 0
tau_xy = T_D*(d/2)/Ip
check("(3) σ_x = 3PLd/(8I)", sigma_x, 3*P*L*d/(8*I))
check("(3) σ_y = 0", sigma_y, 0)
check("(3) τ_xy = PLd/(4 Ip)", tau_xy, P*L*d/(4*Ip))
Ival = sp.pi*d**4/64
Ipval = sp.pi*d**4/32
check("(3) σ_x 数値形 = 24PL/(pi d^3)", sigma_x.subs(I, Ival), 24*P*L/(sp.pi*d**3))
check("(3) τ_xy 数値形 = 8PL/(pi d^3)", tau_xy.subs(Ip, Ipval), 8*P*L/(sp.pi*d**3))

# (4) たわみ・ねじれ角
x = sp.symbols('x')
v = sp.Function('v')
sol = sp.dsolve(sp.Eq(E*I*sp.Derivative(v(x), x, 2), P*(L - x)), v(x),
                ics={v(0): 0, v(x).diff(x).subs(x, 0): 0})
vexpr = sol.rhs
delta_B = vexpr.subs(x, L)
theta_B = sp.diff(vexpr, x).subs(x, L)
check("(4) δ_B = PL^3/(3EI)", delta_B, P*L**3/(3*E*I))
check("(4) θ_B = PL^2/(2EI)", theta_B, P*L**2/(2*E*I))

psi_B = (P*L/2)*L/(G*Ip)
check("(4) ψ_B = PL^2/(2 G Ip)", psi_B, P*L**2/(2*G*Ip))

delta_C = delta_B + psi_B*(L/2) + P*(L/2)**3/(3*E*I)
check("(4) δ_C = 3PL^3/(8EI) + PL^3/(4 G Ip)", sp.simplify(delta_C),
      sp.simplify(3*P*L**3/(8*E*I) + P*L**3/(4*G*Ip)))

note_unverifiable("(I)図/符号", "図からの幾何モデル化（δ=λ1+λ2 の方向）は前提として明記")
note_unverifiable("(II)(1)図示", "反力・モーメント・トルクの図示は描画で対応")
note_unverifiable("(II)(2)SFD/BMD図示", "図は描画で対応")

if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
