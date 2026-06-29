#!/usr/bin/env python3
"""AY2017 材料力学 検算 — 〔I〕複合棒+剛体円盤, 〔II〕直角L字円形断面はり."""
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
A, E, L1, L2, L3, P, d, G, L = sp.symbols('A E L1 L2 L3 P d G L', positive=True)
pi = sp.pi

# ===================== 〔I〕 =====================
# 中央列(棒①+剛体円盤)の自然長 L1+L3 がギャップ L2 に押し込まれる。
# δ1: 棒①の縮み, δ2: 棒②の伸び。 適合: δ1+δ2 = (L1+L3)-L2 = Δ
Delta = L1 + L3 - L2
# 力: 中央圧縮 N1 = 2*N2(②2本の張力合計)。
# δ1 = N1*L1/( (2E)*(2A) )=N1 L1/(4EA),  δ2 = N2*L2/(E*A)
N2 = sp.symbols('N2')
sol = sp.solve(sp.Eq( (2*N2)*L1/(4*E*A) + N2*L2/(E*A), Delta), N2)[0]
N2v = sp.simplify(sol)
N1v = 2*N2v
check("(I) N2 (②各張力)", N2v, 2*E*A*Delta/(L1+2*L2))
check("(I) N1 (①圧縮)", N1v, 4*E*A*Delta/(L1+2*L2))
# 適合確認
d1 = N1v*L1/(4*E*A)
d2 = N2v*L2/(E*A)
check("(I) 適合 δ1+δ2=Δ", sp.simplify(d1+d2), Delta)
# 応力
sig1 = N1v/(2*A)
sig2 = N2v/A
check("(I)(1) σ1 圧縮大きさ", sp.simplify(sig1), 2*E*Delta/(L1+2*L2))
check("(I)(1) σ2 引張", sp.simplify(sig2), 2*E*Delta/(L1+2*L2))
# 変位 λ1=δ1, λ2=δ2
check("(I)(2) λ1 (①縮み)", sp.simplify(d1), Delta*L1/(L1+2*L2))
check("(I)(2) λ2 (②伸び)", sp.simplify(d2), 2*Delta*L2/(L1+2*L2))
# (3) ②を元の長さに戻す(δ2=0 → N2=0)。ギャップ=L2, ①縮み=Δ。
# 上板つり合い: P(下) = N1'(①圧縮 上向き押し) , N1'=Δ*4EA/L1
P_req = Delta*4*E*A/L1
check("(I)(3) P", P_req, 4*E*A*(L1+L3-L2)/L1)

# ===================== 〔II〕 =====================
# 断面: I=pi d^4/64, Ip=pi d^4/32
I = pi*d**4/64
Ip = pi*d**4/32
# AB(x方向,固定A→B), BC(z方向,B→C), C に鉛直下向き P
# B における内力(BCから): せん断P, BC由来の曲げモーメント PL → ABに対しては「ねじり」T=PL
T = P*L
# (1) B点ねじれ角(AB のねじり, 長さL)
PsiB = T*L/(G*Ip)
check("(II)(1) ΨB", sp.simplify(PsiB), 32*P*L**2/(pi*G*d**4))
# (2) vB: AB を先端荷重Pの片持ちはり
vB = P*L**3/(3*E*I)
check("(II)(2) vB", sp.simplify(vB), 64*P*L**3/(3*pi*E*d**4))
# vC = vB(B並進) + ΨB*L(ABねじりでBCが回転) + BC自身の曲げ PL^3/3EI
vC = vB + PsiB*L + P*L**3/(3*E*I)
check("(II)(2) vC 形", sp.simplify(vC), sp.simplify(2*P*L**3/(3*E*I)+P*L**3/(G*Ip)))
check("(II)(2) vC 値", sp.simplify(vC), 128*P*L**3/(3*pi*E*d**4)+32*P*L**3/(pi*G*d**4))
# (3) 固定端A: M=PL(曲げ), T=PL(ねじり), V=P
M = P*L
# 点D(上表面 y=d/2): 曲げ σ=M/Z, ねじりτ=T/Zp, 横せん断=0
Z = pi*d**3/32
Zp = pi*d**3/16
sigD = M/Z
tauD = T/Zp
check("(III) σ_D 曲げ", sp.simplify(sigD), 32*P*L/(pi*d**3))
check("(III) τ_D ねじり", sp.simplify(tauD), 16*P*L/(pi*d**3))
# Dのモール: 中心σ/2, 半径R=sqrt((σ/2)^2+τ^2)
R_D = sp.sqrt((sigD/2)**2 + tauD**2)
tmaxD = R_D
s1D = sigD/2 + R_D
s2D = sigD/2 - R_D
check("(III) τmax_D", sp.simplify(tmaxD), 16*sp.sqrt(2)*P*L/(pi*d**3))
check("(III) σ1_D", sp.simplify(s1D), 16*(1+sp.sqrt(2))*P*L/(pi*d**3))
check("(III) σ2_D", sp.simplify(s2D), 16*(1-sp.sqrt(2))*P*L/(pi*d**3))
# 点E(中立面 y=0): σ=0, τ=ねじり(横せん断はL>>dで無視)。純せん断。
sigE = 0
tauE = tauD
check("(III) σ_E", sigE, 0)
check("(III) τ_E", sp.simplify(tauE), 16*P*L/(pi*d**3))
tmaxE = sp.sqrt((sigE/2)**2 + tauE**2)
check("(III) τmax_E", sp.simplify(tmaxE), 16*P*L/(pi*d**3))
check("(III) σ1_E", sp.simplify(tauE), 16*P*L/(pi*d**3))
check("(III) σ2_E", sp.simplify(-tauE), -16*P*L/(pi*d**3))
# 参考: 横せん断(中立軸,円形 4V/3A)はPL/d^3に対しd/L倍で小さい
note_unverifiable("(III) E点横せん断省略", "L>>d より τ_V~P/d^2 はねじり τ~PL/d^3 の d/L 倍で無視")

if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
