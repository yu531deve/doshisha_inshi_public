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


# === 各小問の検算 =========================================
P, E, A, L = sp.symbols('P E A L', positive=True)

# --- 〔1〕 組合せ棒 ---
# 部材剛性 k = (断面積*縦弾性係数)/長さ
k1 = (3*A)*(2*E)/(L/2)        # 丸棒1: 3A, 2E, L/2  -> 12AE/L
k2 = (A)*(E)/(L/2)            # 丸棒2: A, E, L/2     -> 2AE/L
k3 = (4*A)*(5*E)/(L)          # 円管3: 4A, 5E, L     -> 20AE/L
check("(1) k1=12AE/L", k1, 12*A*E/L)
check("(1) k2=2AE/L",  k2, 2*A*E/L)
check("(1) k3=20AE/L", k3, 20*A*E/L)

# 内側経路(丸棒1+丸棒2 直列)
k_inner = 1/(1/k1 + 1/k2)
check("(1) k_inner=12AE/(7L)", sp.simplify(k_inner), 12*A*E/(7*L))

# プレート変位 δ (内側経路と円管3は並列、同一変位)
delta = P/(k_inner + k3)
check("(1) δ=7PL/(152AE)", sp.simplify(delta), 7*P*L/(152*A*E))

N_inner = sp.simplify(k_inner*delta)
N3 = sp.simplify(k3*delta)
check("(1) N1=N2=3P/38", N_inner, sp.Rational(3,38)*P)
check("(1) N3=35P/38",  N3, sp.Rational(35,38)*P)
check("(1) 全体つり合い N_inner+N3=P", N_inner+N3, P)

N1 = N_inner; N2 = N_inner
sig1 = sp.simplify(N1/(3*A)); sig2 = sp.simplify(N2/A); sig3 = sp.simplify(N3/(4*A))
check("(1) σ1=P/(38A)",  sig1, P/(38*A))
check("(1) σ2=3P/(38A)", sig2, 3*P/(38*A))
check("(1) σ3=35P/(152A)", sig3, 35*P/(152*A))

lam1 = sp.simplify(N1*(L/2)/((3*A)*(2*E)))
lam2 = sp.simplify(N2*(L/2)/((A)*(E)))
lam3 = sp.simplify(N3*(L)/((4*A)*(5*E)))
check("(1) λ1=PL/(152AE)",  lam1, P*L/(152*A*E))
check("(1) λ2=3PL/(76AE)",  lam2, 3*P*L/(76*A*E))
check("(1) λ3=7PL/(152AE)", lam3, 7*P*L/(152*A*E))
check("(1) λ1+λ2=δ", sp.simplify(lam1+lam2), delta)
check("(1) λ3=δ",    lam3, delta)

# --- (2) 数値 ---
subs = {P:200e3, E:100e9, A:100e-6, L:1.0}   # SI: N, Pa, m^2, m
PL_AE = float((P*L/(A*E)).subs(subs))         # = 0.02 m = 20 mm
check("(2) PL/AE=0.02m", PL_AE, 0.02)
check("(2) λ1≈0.1316mm", float(lam1.subs(subs))*1e3, 20/152, tol=1e-6)
check("(2) λ2≈0.7895mm", float(lam2.subs(subs))*1e3, 60/76, tol=1e-6)
check("(2) λ3≈0.9211mm", float(lam3.subs(subs))*1e3, 140/152, tol=1e-6)

# --- (3) X点に外力F、丸棒1の応力0 (N1=0 → u_X=0) ---
# u_X: X点変位, dtop: プレート変位
uX, dtop = sp.symbols('u_X d_top')
N1f = k1*uX
N2f = k2*(dtop - uX)
N3f = k3*dtop
# N1=0 -> uX=0
sol = sp.solve([sp.Eq(N1f, 0), sp.Eq(N2f + N3f, P)], [uX, dtop], dict=True)[0]
uXv = sol[uX]; dtopv = sol[dtop]
check("(3) u_X=0", uXv, 0)
check("(3) d_top=PL/(22AE)", sp.simplify(dtopv), P*L/(22*A*E))
N2v = sp.simplify(N2f.subs(sol)); N3v = sp.simplify(N3f.subs(sol))
check("(3) N2=P/11", N2v, P/11)
check("(3) N3=10P/11", N3v, 10*P/11)
# X点つり合い: N2 - N1 + F = 0 -> F = N1 - N2 (上向き正)
F = sp.simplify(0 - N2v)
check("(3) F=-P/11 (下向き P/11)", F, -P/11)
sig2_3 = sp.simplify(N2v/A); sig3_3 = sp.simplify(N3v/(4*A))
check("(3) σ2=P/(11A)", sig2_3, P/(11*A))
check("(3) σ3=5P/(22A)", sig3_3, 5*P/(22*A))
lam2_3 = sp.simplify(N2v*(L/2)/((A)*(E)))
lam3_3 = sp.simplify(N3v*(L)/((4*A)*(5*E)))
check("(3) λ2=PL/(22AE)", lam2_3, P*L/(22*A*E))
check("(3) λ3=PL/(22AE)", lam3_3, P*L/(22*A*E))
# (3) 数値
check("(3) F≈18.18kN", float((-F).subs(subs))/1e3, 200/11, tol=1e-6)
check("(3) λ2≈0.909mm", float(lam2_3.subs(subs))*1e3, 20/22, tol=1e-6)
check("(3) σ2≈181.8MPa", float(sig2_3.subs(subs))/1e6, (200e3/11)/(100e-6)/1e6, tol=1e-3)
check("(3) σ3≈454.5MPa", float(sig3_3.subs(subs))/1e6, (5*200e3/22)/(100e-6)/1e6, tol=1e-3)

# --- 〔2〕 薄肉円筒 内圧 p, 肉厚 t, 径 d ---
p, t, d = sp.symbols('p t d', positive=True)
sig_theta = p*d/(2*t)
sig_x = p*d/(4*t)
# フープ: 単位長さ つり合い σθ*2t = p*d
check("〔2〕 σθ*2t = p d", sp.simplify(sig_theta*2*t - p*d), 0)
# 軸方向: σx*(π d t) = p*(π d^2/4)
check("〔2〕 σx*(π d t) = p(π d^2/4)",
      sp.simplify(sig_x*(sp.pi*d*t) - p*(sp.pi*d**2/4)), 0)
check("〔2〕 σθ=2 σx", sig_theta, 2*sig_x)

# --- 〔3〕 曲げ＋ねじり（円形断面はり, 先端剛体腕にF） ---
F, a = sp.symbols('F a', positive=True)
# 断面定数（中実丸棒）
I  = sp.pi*d**4/64     # 断面二次モーメント
Ip = sp.pi*d**4/32     # 断面二次極モーメント
# 点Q（自由端から距離a）の曲げモーメント M=F a, ねじりトルク T=F(L/2)
M = F*a
T = F*(L/2)
# (1) 曲げ応力 σx = M(d/2)/I, 円周方向 σy=0
sigx3 = sp.simplify(M*(d/2)/I)
check("〔3〕(1) σx=32Fa/(πd^3)", sigx3, 32*F*a/(sp.pi*d**3))
sigy3 = 0
check("〔3〕(1) σy=0", sigy3, 0)
# (2) せん断応力 τxy = T(d/2)/Ip （最外縁なので横せん断は0, ねじりのみ）
tau3 = sp.simplify(T*(d/2)/Ip)
check("〔3〕(2) τxy=8FL/(πd^3)", tau3, 8*F*L/(sp.pi*d**3))
# (3) モール円: 中心 σx/2, 半径 R, 主応力, τmax
center = sp.simplify(sigx3/2)
check("〔3〕(3) 中心 σx/2=16Fa/(πd^3)", center, 16*F*a/(sp.pi*d**3))
R = sp.sqrt((sigx3/2)**2 + tau3**2)
R_closed = 8*F*sp.sqrt(4*a**2 + L**2)/(sp.pi*d**3)
check("〔3〕(3) R=8F√(4a²+L²)/(πd^3)", sp.simplify(R - R_closed), 0)
sig1 = sp.simplify(center + R)
sig2 = sp.simplify(center - R)
taumax = sp.simplify(R)
sig_max_closed = (8*F/(sp.pi*d**3))*(2*a + sp.sqrt(4*a**2 + L**2))
check("〔3〕(3) σmax=σ1=(8F/πd^3)(2a+√(4a²+L²))",
      sp.simplify(sig1 - sig_max_closed), 0)
check("〔3〕(3) τmax=R", sp.simplify(taumax - R_closed), 0)
# σ1+σ2 = σx (不変量), σ1*σ2 = -τxy² (σy=0)
check("〔3〕(3) σ1+σ2=σx", sp.simplify(sig1 + sig2), sigx3)
check("〔3〕(3) σ1 σ2 = -τxy²", sp.simplify(sig1*sig2 + tau3**2), 0)
# (4) 薄肉円筒置換: Iz, Ip を残した一般形
Iz, Ip2 = sp.symbols('I_z I_p', positive=True)
sigx4 = sp.simplify(M*(d/2)/Iz)
tau4  = sp.simplify(T*(d/2)/Ip2)
check("〔3〕(4) σx=Fad/(2Iz)", sigx4, F*a*d/(2*Iz))
check("〔3〕(4) σy=0", 0, 0)
check("〔3〕(4) τxy=FLd/(4Ip)", tau4, F*L*d/(4*Ip2))
# 中実丸棒の Iz=I, Ip=Ip を代入すると(1)(2)に一致
check("〔3〕(4)->(1) 整合", sp.simplify(sigx4.subs(Iz, I)), sigx3)
check("〔3〕(4)->(2) 整合", sp.simplify(tau4.subs(Ip2, Ip)), tau3)
# モール円作図は数値検算不能
note_unverifiable("〔3〕(3) モール円の作図", "tikzによる作図（数値検算の対象外）")
# =========================================================


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
