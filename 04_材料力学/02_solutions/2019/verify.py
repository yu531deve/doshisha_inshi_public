#!/usr/bin/env python3
"""AY2019 材料力学 検算。全 PASS にしてから tex を書く。"""
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


P, L, A, E, alpha, dT = sp.symbols('P L A E alpha Delta_T', positive=True)
M0, I, W, x = sp.symbols('M0 I W x', positive=True)

# ===== 〔I〕 不静定棒（外力P + 温度上昇ΔT）=====
# b点変位 u（右向き正）を未知数に。a固定(x=0), c固定(x=2L)。
# 棒1: a-b, (A,E,α)  伸び λ1 = N1 L/(EA) + αLΔT = u
# 棒2: b-c, (A,2E,2α) 伸び λ2 = N2 L/(2EA) + 2αLΔT = -u  (b右移動でc固定→縮む)
# 節点b つり合い(右向き正): P - N1 + N2 = 0  → N1 - N2 = P
u, N1, N2 = sp.symbols('u N1 N2')
eq1 = sp.Eq(N1*L/(E*A) + alpha*L*dT, u)
eq2 = sp.Eq(N2*L/(2*E*A) + 2*alpha*L*dT, -u)
eq3 = sp.Eq(N1 - N2, P)
sol = sp.solve([eq1, eq2, eq3], [u, N1, N2], dict=True)[0]
N1s, N2s, us = sol[N1], sol[N2], sol[u]

sig1 = sp.simplify(N1s/A)
sig2 = sp.simplify(N2s/A)
check("I(1) σ1", sig1, P/(3*A) - 2*E*alpha*dT)
check("I(1) σ2", sig2, -2*P/(3*A) - 2*E*alpha*dT)
# 平衡再確認
check("I(1) 平衡 N1-N2=P", sp.simplify(N1s - N2s), P)

lam1 = sp.simplify(us)               # 棒1の伸び = b点変位
lam2 = sp.simplify(-us)              # 棒2の変形
check("I(2) λ1", lam1, P*L/(3*E*A) - alpha*L*dT)
check("I(2) λ2", lam2, -P*L/(3*E*A) + alpha*L*dT)
check("I(2) 適合 λ1+λ2=0", sp.simplify(lam1 + lam2), 0)

# ===== 〔II〕単純支持はり span=2L =====
# (1) 両端モーメント → 一定曲げ。図の向きは中央が上に凸（hogging）。
#     EI v'' = M(x) = -M0  （上に凸に対応）, BC v(0)=v(2L)=0
C1, C2 = sp.symbols('C1 C2')
vp = sp.integrate(-M0, x) + C1          # EI v'
v  = sp.integrate(vp, x) + C2           # EI v
cc = sp.solve([v.subs(x, 0), v.subs(x, 2*L)], [C1, C2])
vEI = sp.expand(v.subs(cc))             # = EI v
vpEI = sp.expand(vp.subs(cc))           # = EI v'
check("II(1) EI v(x)", vEI, M0*(2*L*x - x**2)/2)
check("II(1) EI v'(x)", vpEI, M0*(L - x))
check("II(1) たわみ角 θ_A", vpEI.subs(x, 0), M0*L)
check("II(1) たわみ角 θ_B", vpEI.subs(x, 2*L), -M0*L)
vC_M0 = vEI.subs(x, L)
check("II(1) 中央たわみ EI*v_C", vC_M0, M0*L**2/2)   # 上向き(正)

# (2) 中央集中荷重 W(下向き) による中央たわみ = -W(2L)^3/(48EI)
#     合計たわみ=0: v_C(M0) + v_C(W) = 0
vC_W_EI = -W*(2*L)**3/48
Wsol = sp.solve(sp.Eq(vC_M0 + vC_W_EI, 0), W)[0]
check("II(2) W", Wsol, 3*M0/L)
# 数値検算
subs = {M0: 5.0, L: 2.0, E: 1.0, I: 1.0}
EI = 1.0
vM = float((M0*L**2/2).subs(subs))/EI
vW = float((-Wsol*(2*L)**3/48).subs(subs))/EI
check("II(2) 数値: 中央たわみ合計=0", vM + vW, 0.0, tol=1e-9)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
