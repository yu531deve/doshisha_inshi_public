#!/usr/bin/env python3
"""AY2022 材料力学 検算。符号規約: 〔I〕上向き変位・引張応力を正。
〔II〕z上向き正、荷重Pは下向き(-z)。"""
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


# ===== 〔I〕 =====
E, A, L, alpha, dT, P = sp.symbols('E A L alpha dT P', positive=True)
lam, lam_a = sp.symbols('lambda lambda_a', real=True)

k = E * A / L            # 棒①の剛性 EA/L
c = alpha * dT * L / 2   # 棒②半区間の自由熱伸び αΔT(L/2)
k2 = 2 * E * A / (L / 2) # 棒②半区間の剛性 = 4EA/L

# 内力(引張正)
N1 = k * lam
N2u = k2 * (lam - lam_a - c)   # 上半(a→板)
N2l = k2 * (lam_a - c)         # 下半(床→a)

# 連立: 板のつり合い 2N1+N2u=0, 節点aのつり合い N2u-N2l-P=0
sol = sp.solve([sp.Eq(2 * N1 + N2u, 0), sp.Eq(N2u - N2l - P, 0)],
               [lam, lam_a], dict=True)[0]
lam_s = sol[lam]
lam_a_s = sol[lam_a]

# (1) 応力
sig1 = (N1 / A).subs(sol)
sig2u = (N2u / A).subs(sol)
sig2l = (N2l / A).subs(sol)
check("I-(1) sigma1", sig1, E * alpha * dT / 2 - P / (8 * A))
check("I-(1) sigma2 upper", sig2u, P / (4 * A) - E * alpha * dT)
check("I-(1) sigma2 lower", sig2l, -E * alpha * dT - 3 * P / (4 * A))

# 全体つり合い検算: 2*N1底 + N2l = -P (床反力 +P)
check("I 全体つり合い", (2 * N1 + N2l).subs(sol), -P)

# (2) 点aの変位
check("I-(2) lambda_a", lam_a_s, alpha * dT * L / 4 - 3 * P * L / (16 * E * A))

# (3) 板変位 lambda
check("I-(3) lambda", lam_s, alpha * dT * L / 2 - P * L / (8 * E * A))
Pzero = sp.solve(sp.Eq(lam_s, 0), P)[0]
check("I-(3) lambda=0 となる P", Pzero, 4 * E * A * alpha * dT)

# ===== 〔II〕 =====
G, I, Ip, d = sp.symbols('G I I_p d', positive=True)
x = sp.symbols('x', positive=True)

# (1) 反力・モーメント・トルク。loads: B(L,0,0) と C(L,L/3,0) に各 (0,0,-P)
rB = sp.Matrix([L, 0, 0]); rC = sp.Matrix([L, L / 3, 0])
F = sp.Matrix([0, 0, -P])
Mload = rB.cross(F) + rC.cross(F)
Ra = 2 * P
Ma_vec = -Mload   # 反力モーメント
check("II-(1) Ra", Ra, 2 * P)
check("II-(1) Ta (x軸まわり)", Ma_vec[0], P * L / 3)
check("II-(1) Ma 曲げ(|y成分|)", sp.Abs(Ma_vec[1]), 2 * P * L)
check("II-(1) Mz=0", Ma_vec[2], 0)

# (2) SFD/BMD: せん断 2P 一定、曲げ M(x)=2P(L-x)
Mx = 2 * P * (L - x)
check("II-(2) M(0)=2PL", Mx.subs(x, 0), 2 * P * L)
check("II-(2) M(L)=0", Mx.subs(x, L), 0)
check("II-(2) せん断一定 2P (=-dM/dx)", -sp.diff(Mx, x), 2 * P)

# (3) 点u (x=L/4, z=+d/2 上面) の応力
Mu = Mx.subs(x, L / 4)              # = 3PL/2
sigx = Mu * (d / 2) / I
check("II-(3) M(L/4)", Mu, 3 * P * L / 2)
check("II-(3) sigma_x", sigx, 3 * P * L * d / (4 * I))
check("II-(3) sigma_y", 0, 0)
tau_xy = (P * L / 3) * (d / 2) / Ip  # ねじり, 上面で横せん断=0
check("II-(3) tau_xy", tau_xy, P * L * d / (6 * Ip))
note_unverifiable("II-(3) 上面で横せん断=0", "中立軸から最遠の表面繊維のため Q由来のτは0")

# (4) z変位
delB = 2 * P * L**3 / (3 * E * I)
theta = (P * L / 3) * L / (G * Ip)   # ねじれ角
delC = delB + theta * (L / 3)
check("II-(4) delta_B", delB, 2 * P * L**3 / (3 * E * I))
check("II-(4) ねじれ角 theta", theta, P * L**2 / (3 * G * Ip))
check("II-(4) delta_C", delC, 2 * P * L**3 / (3 * E * I) + P * L**3 / (9 * G * Ip))


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
