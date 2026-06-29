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


# === 記号 ===
P, L, S, E, alpha, dT = sp.symbols('P L S E alpha dT', positive=True)
f0, d, G, x, Q = sp.symbols('f0 d G x Q', positive=True)

# ===================== 〔I〕 =====================
# 板の右向き変位 u を未知数に。棒①: 左壁-板, 長2L,断面2S,弾性E(伸び u)
#                              棒②: 板-右壁, 長L,断面S,弾性3E(縮み u)
# 棒① 軸力(引張+) N1 = 2S*E * (u/(2L)) = S*E*u/L
# 棒② 伸び = -u(幾何) = 機械的伸び + 熱伸び
#      N2*L/(3S*E) + 4*alpha*dT*L = -u  → N2 = -3*S*E*u/L - 12*S*E*alpha*dT
# 板つり合い(右向き+): P - N1 + N2 = 0
u = sp.symbols('u')
N1 = S*E*u/L
N2 = -3*S*E*u/L - 12*S*E*alpha*dT

# --- (1)(2): 温度なし dT=0 ---
sol_u = sp.solve(sp.Eq(P - N1.subs(dT,0) + N2.subs(dT,0), 0), u)[0]
check("(1) u=PL/4SE", sol_u, P*L/(4*S*E))
N1_0 = N1.subs({dT:0, u:sol_u}); N2_0 = N2.subs({dT:0, u:sol_u})
check("(1) sigma1 = P/8S (引張)", N1_0/(2*S), P/(8*S))
check("(1) sigma2 = -3P/4S (圧縮)", N2_0/S, -3*P/(4*S))
# (2) 変位 λ1=棒①伸び=u, λ2=棒②縮み=u
lam1 = sol_u                      # 棒① elongation = u
lam2 = sol_u                      # 棒② shortening = u
check("(2) lambda1 = PL/4SE", lam1, P*L/(4*S*E))
check("(2) lambda2 = PL/4SE", lam2, P*L/(4*S*E))
# 個別に応力から確認
check("(2) lambda1 via N1", N1_0*(2*L)/(2*S*E), P*L/(4*S*E))
check("(2) lambda2 via N2", sp.Abs(N2_0)*(L)/(3*S*E), P*L/(4*S*E))

# --- (3)(4): 棒②加熱, 各棒が初期長さに戻る → u=0 ---
N1_h = N1.subs(u,0); N2_h = N2.subs(u,0)
check("(3) N1' = 0", N1_h, 0)
sol_dT = sp.solve(sp.Eq(P - N1_h + N2_h, 0), dT)[0]
check("(3) dT = P/(12 S E alpha)", sol_dT, P/(12*S*E*alpha))
N2_h_val = N2_h.subs(dT, sol_dT)
check("(4) sigma1' = 0", N1_h/(2*S), 0)
check("(4) sigma2' = -P/S", N2_h_val/S, -P/S)

# ===================== 〔II〕 =====================
I = sp.pi*d**4/64
J = sp.pi*d**4/32
# (1) 支持反力(固定端B, x=2Lにある)
RB = sp.integrate(f0, (x,0,2*L))
check("(1) R_B = 2 f0 L", RB, 2*f0*L)
MB = f0*(2*L)*(L)   # 合力 f0*2L が中央(Bから距離L)
check("(1) M_B = 2 f0 L^2", MB, 2*f0*L**2)
note_unverifiable("(1) T_B = T_A", "ねじりトルクは自由端から固定端まで一定で伝達(図読み取り)")
# (2) Aから x の断面: 自由端側 0..x の分布荷重 f0, 合力 f0*x が断面から x/2
s_ = sp.symbols('s_', positive=True)
M_int = sp.integrate(f0*(x - s_), (s_, 0, x))   # ∫f0*(腕長) ds
Mx = f0*x**2/2
check("(2) M(x) 積分一致", sp.simplify(M_int - Mx), 0)
check("(2) M(x) = f0 x^2/2", Mx, f0*x*(x/2))
note_unverifiable("(2) T(x) = T_A", "全長一定のねじりトルク")
# (3) 点C (x=L) 上面
MC = Mx.subs(x, L)
check("(3) M_C = f0 L^2/2", MC, f0*L**2/2)
sigx = MC*(d/2)/I
check("(3) sigma_x = 16 f0 L^2/(pi d^3)", sp.simplify(sigx), 16*f0*L**2/(sp.pi*d**3))
TA = sp.symbols('T_A', positive=True)
tau = TA*(d/2)/J
check("(3) tau_xy = 16 T_A/(pi d^3)", sp.simplify(tau), 16*TA/(sp.pi*d**3))
note_unverifiable("(3) sigma_y = 0", "単軸曲げ, 横方向垂直応力なし")
# (4) たわみ δ_A: 基礎式 EI v'' = M. x:A(0)..B(2L), M(x)=f0 x^2/2(片持はり)
v = sp.symbols('v')
C1, C2 = sp.symbols('C1 C2')
EI = sp.symbols('EI', positive=True)
vpp = Mx/EI
vp = sp.integrate(vpp, x) + C1
vfun = sp.integrate(vp, x) + C2
# 固定端 x=2L: v=0, v'=0
sol_c = sp.solve([vfun.subs(x,2*L), vp.subs(x,2*L)], [C1,C2])
v_full = vfun.subs(sol_c)
deltaA = sp.simplify(v_full.subs(x,0))
check("(4) delta_A = 2 f0 L^4/(EI)", deltaA, 2*f0*L**4/EI)
check("(4) delta_A (Id代入) = 128 f0 L^4/(pi E d^4)",
      deltaA.subs(EI, E*I), 128*f0*L**4/(sp.pi*E*d**4))
phi = TA*(2*L)/(G*J)
check("(4) phi = 64 L T_A/(pi G d^4)", sp.simplify(phi), 64*L*TA/(sp.pi*G*d**4))
# (5) カスチリアーノ: 仮想集中荷重 Q を A に。M = f0 x^2/2 + Q x, dM/dQ = x
Mq = f0*x**2/2 + Q*x
dMdQ = sp.diff(Mq, Q)
deltaA_cas = sp.integrate((Mq.subs(Q,0)*dMdQ.subs(Q,0))/EI, (x,0,2*L))
check("(5) Castigliano delta_A = 2 f0 L^4/(EI)", sp.simplify(deltaA_cas), 2*f0*L**4/EI)
check("(5) Castigliano == 基礎式", sp.simplify(deltaA_cas - deltaA), 0)


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
