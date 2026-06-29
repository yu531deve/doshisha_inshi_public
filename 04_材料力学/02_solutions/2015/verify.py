#!/usr/bin/env python3
"""AY2015 材料力学 検算"""
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

P, L, d, E, G, x, f0, b, h = sp.symbols('P L d E G x f0 b h', positive=True)

# =========================================================
# 〔I〕 片持ちはり + 剛体棒(長さL/2) 先端に鉛直下向きP
# 等価系: 自由端Aに 下向きP と 軸まわりトルク T=P*L/2
# =========================================================
R_B = P; M_B = P*L; T_B = P*L/2
check("(I-1) 反力 R_B", R_B, P)
check("(I-1) 曲げ反モーメント M_B", M_B, P*L)
check("(I-1) 反トルク T_B", T_B, P*L/2)

Q = P; M = P*x
check("(I-2) Q", Q, P)
check("(I-2) M(0)=0", M.subs(x,0), 0)
check("(I-2) M(L)=PL", M.subs(x,L), P*L)

I = sp.pi*d**4/64; J = sp.pi*d**4/32; c = d/2
sigma = (P*L)*c/I
tau_t = T_B*c/J
check("(I-3) 曲げ応力 σ", sigma, 32*P*L/(sp.pi*d**3))
check("(I-3) せん断応力 τ(ねじり)", tau_t, 8*P*L/(sp.pi*d**3))
note_unverifiable("(I-3) 横せん断", "最外縁Cでは横せん断応力=0。Cはねじりのみ")

sx, sy, txy = sigma, 0, tau_t
cen = (sx+sy)/2
rad = sp.sqrt(((sx-sy)/2)**2 + txy**2)
s1 = sp.simplify(cen+rad); s2 = sp.simplify(cen-rad); tmax = sp.simplify(rad)
k = P*L/(sp.pi*d**3)
check("(I-4) 主応力 σ1", s1, (16+8*sp.sqrt(5))*k)
check("(I-4) 主応力 σ2", s2, (16-8*sp.sqrt(5))*k)
check("(I-4) 最大せん断 τmax", tmax, 8*sp.sqrt(5)*k)

# =========================================================
# 〔II〕 突き出しはり: A固定(x=0), Bローラー(x=L), C自由端(x=2L)
# AB間 等分布 f0(下), C点 集中P(下)。1次不静定。
# 符号: 上向き反力正, さがり曲げ正, EI v''=M
# =========================================================
M0, RA, RB, C3, C4 = sp.symbols('M0 RA RB C3 C4')
M1 = M0 + RA*x - f0*x**2/2
M2 = M0 + RA*x - f0*L*(x - L/2) + RB*(x-L)

EIv1p = sp.integrate(M1, x)          # v1(0)=0,v1'(0)=0 → C1=C2=0
EIv1  = sp.integrate(EIv1p, x)
EIv2p = sp.integrate(M2, x) + C3
EIv2  = sp.integrate(EIv2p, x) + C4

eqs = [
    sp.Eq(EIv1.subs(x,L), 0),
    sp.Eq(EIv2.subs(x,L), 0),
    sp.Eq(EIv1p.subs(x,L), EIv2p.subs(x,L)),
    sp.Eq(RA+RB - f0*L - P, 0),
    sp.Eq(M2.subs(x,2*L), 0),
]
sol = sp.solve(eqs, [M0, RA, RB, C3, C4], dict=True)[0]
RA_v = sp.simplify(sol[RA]); RB_v = sp.simplify(sol[RB]); M0_v = sp.simplify(sol[M0])
print("R_A=",RA_v," R_B=",RB_v," M0(=内力 at A)=",M0_v," C3=",sp.simplify(sol[C3])," C4=",sp.simplify(sol[C4]))

check("(II) 力つり合い", RA_v+RB_v, f0*L+P)
# 反力カップル M_A(さがり正の内力 M0)。Aまわりモーメントつり合い:
#   -M0(内力符号) + R_B*L - f0*L*(L/2) - P*2L = 0
MA_couple = -M0_v   # 壁の反モーメント(ホギング正)
check("(II) Aまわりモーメントつり合い", sp.simplify(MA_couple + RB_v*L - f0*L**2/2 - P*2*L), 0)

EIvC = sp.simplify(EIv2.subs(sol).subs(x,2*L))
print("EI*v_C =", EIvC)
check("(II) v(L)=0 ローラー", sp.simplify(EIv1.subs(sol).subs(x,L)), 0)
check("(II) M(2L)=0 自由端", sp.simplify(M2.subs(sol).subs(x,2*L)), 0)

# 閉形(sympy導出)との照合
check("(II) R_A", RA_v, sp.Rational(5,8)*f0*L - sp.Rational(3,2)*P)
check("(II) R_B", RB_v, sp.Rational(3,8)*f0*L + sp.Rational(5,2)*P)
check("(II) M_A(壁反モーメント,ホギング)", MA_couple, f0*L**2/8 - P*L/2)
check("(II) EI v_C", EIvC, f0*L**4/48 - sp.Rational(7,12)*P*L**3)

# 検算: f0のみ(P=0)→標準プロップド片持ち R_B=3f0L/8, M_A=f0L^2/8
check("(II検算) P=0 で R_B=3f0L/8", RB_v.subs(P,0), sp.Rational(3,8)*f0*L)
check("(II検算) P=0 で M_A=f0L^2/8", MA_couple.subs(P,0), f0*L**2/8)
# 検算: f0=0(C先端Pのみ,プロップド片持ち+オーバーハング)→ R_B=5P/2 上向き
check("(II検算) f0=0 で R_B=5P/2", RB_v.subs(f0,0), sp.Rational(5,2)*P)

if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
