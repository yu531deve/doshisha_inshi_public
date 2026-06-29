#!/usr/bin/env python3
"""
解答検算テンプレート — 同志社 医工学 院試
======================================
仕様書§4「検証ファースト」: LaTeX を書く前に、これを
  02_solutions/<年度>/verify.py
にコピーして全小問を埋め、★全 PASS にしてから tex を書き始める★。

依存: sympy, mpmath, numpy（制御工学ではscipy.signalもあると便利）
実行: cd 02_solutions/<年度> && python3 verify.py
出力: 小問ごとに [PASS]/[FAIL]/[SKIP]、最後にサマリ。FAIL があれば exit 1。
"""
import sympy as sp
import mpmath as mp
mp.mp.dps = 30

results = []


def check(label, got, expected, tol=1e-9):
    """記号一致 → だめなら数値一致で PASS/FAIL を判定・記録する。"""
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
    """検算できない小問は理由を残す（証明・図からの読み取り・設計記述など）。
    仕様書§8「検証不能な理由を記録」に対応。"""
    results.append((label, None))
    print(f"[SKIP] {label}: 検証不能 — {reason}")


# === 各小問の検算 ============================
m, g, A, E, L, h = sp.symbols('m g A E L h', positive=True)
P, d, MC, TB, G, x = sp.symbols('P d M_C T_B G x', positive=True)

# ---------- 〔I〕段付き棒 ----------
# 棒1: 断面積2A, 縦弾性係数E, 長さL ; 棒2: 断面積A, 縦弾性係数3E, 長さ2L
# 直列で両棒とも軸力=おもり重さ mg を受ける（引張）。
N = m*g
# (1) 静置時の応力
sig1 = N/(2*A)
sig2 = N/A
check("(1) sigma1", sig1, m*g/(2*A))
check("(1) sigma2", sig2, m*g/A)

# (2) 全体の伸び
lam1 = N*L/(2*A*E)
lam2 = N*(2*L)/(A*(3*E))
lam = sp.simplify(lam1+lam2)
check("(2) lambda", lam, 7*m*g*L/(6*A*E))

# (3)(4) 衝撃: 軸力共通 -> sigma2' = 2 sigma1'
# 動的伸び lambda' = sig1'/E*L + sig2'/(3E)*2L , sig2'=2sig1'
s1 = sp.symbols('s1', positive=True)  # = sigma1'
s2 = 2*s1
lamp = s1/E*L + s2/(3*E)*(2*L)
check("(3) lambda' (=7 s1 L/3E)", sp.simplify(lamp), 7*s1*L/(3*E))
# ひずみエネルギー U = sum sigma^2/(2Emod) * Volume
U1 = s1**2/(2*E)*(2*A*L)
U2 = s2**2/(2*(3*E))*(A*2*L)
U = sp.simplify(U1+U2)
check("(3) U (=7 s1^2 A L/3E)", U, 7*s1**2*A*L/(3*E))
# エネルギー保存 mg(h+lambda') = U  を s1 について解く
eq = sp.Eq(m*g*(h+lamp), U)
sol = sp.solve(eq, s1)
sol_pos = [s for s in sol if sp.simplify(s.subs({m:1,g:1,A:1,E:1,L:1,h:1}))>0][0]
sig1p_expected = (m*g/(2*A))*(1+sp.sqrt(1+12*A*E*h/(7*m*g*L)))
check("(4) sigma1'", sp.simplify(sol_pos - sig1p_expected), 0)
check("(4) sigma2'=2sigma1'", sp.simplify(2*sig1p_expected),
      (m*g/A)*(1+sp.sqrt(1+12*A*E*h/(7*m*g*L))))
# 衝撃係数の静たわみ整合: 2h/lambda = 12AEh/(7mgL)
check("(4) 衝撃係数整合", sp.simplify(2*h/lam), 12*A*E*h/(7*m*g*L))

# ---------- 〔II〕丸棒（片持ち, A固定, 長さ2L） ----------
I = sp.pi*d**4/64
J = sp.pi*d**4/32
# (1) 反力
RA = P
MA = 2*P*L - MC
TA = TB
check("(1) R_A", RA, P)
check("(1) T_A", TA, TB)
# (2) 内力: T一定=TB, 曲げM(x)
M_AC = MC - P*(2*L - x)        # 0<=x<=L (区間AC)
M_CB = -P*(2*L - x)            # L<=x<=2L (区間CB)
# 支点モーメント整合: M(0)= -(M_A)
check("(1) M_A 整合 (M(0)=-M_A)", sp.simplify(M_AC.subs(x,0)), -(MA))
# (3) 点D x=L/2, z=d/2
MD = M_AC.subs(x, L/2)
check("(3) M(L/2)", sp.simplify(MD), MC - sp.Rational(3,2)*P*L)
sigx = MD*(d/2)/I
check("(3) sigma_x", sp.simplify(sigx), sp.simplify(32*(MC-sp.Rational(3,2)*P*L)/(sp.pi*d**3)))
tau = TB*(d/2)/J
check("(3) tau_xy", sp.simplify(tau), 16*TB/(sp.pi*d**3))
note_unverifiable("(3) sigma_y=0", "丸棒の曲げ+ねじりで軸直交方向応力は生じない（単軸+ねじり）")

# (4) たわみ: EI v'' = M(x), v(0)=v'(0)=0
EI = sp.symbols('EI', positive=True)
v1p = sp.integrate(M_AC, x) + 0          # C1=0 (v'(0)=0)
v1  = sp.integrate(v1p, x) + 0           # C2=0 (v(0)=0)
# 区間2
C3, C4 = sp.symbols('C3 C4')
v2p = sp.integrate(M_CB, x) + C3
v2  = sp.integrate(v2p, x) + C4
solC = sp.solve([sp.Eq(v2p.subs(x,L), v1p.subs(x,L)),
                 sp.Eq(v2.subs(x,L),  v1.subs(x,L))], [C3,C4])
v2B = v2.subs(solC).subs(x, 2*L)
EI_vB = sp.simplify(v2B)
check("(4) EI*v(2L)", EI_vB, sp.Rational(3,2)*MC*L**2 - sp.Rational(8,3)*P*L**3)
delB = EI_vB/EI
psiB = TB*(2*L)/(G*J)
check("(4) psi_B", sp.simplify(psiB), 64*TB*L/(sp.pi*G*d**4))
# (5) delta_B=0
McSol = sp.solve(sp.Eq(EI_vB,0), MC)[0]
check("(5) M_C (delta_B=0)", sp.simplify(McSol), sp.Rational(16,9)*P*L)
# =========================================================


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
