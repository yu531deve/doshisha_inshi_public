#!/usr/bin/env python3
"""
AY2012 力学 解答検算 — 同志社 医工学 院試
〔I〕ばね連成系（2質点・1次元）、〔II〕弾性衝突（運動量・エネルギー保存）。
実行: cd 02_solutions/2012 && python3 verify.py
"""
import sympy as sp

results = []


def check(label, got, expected):
    ok = sp.simplify(sp.sympify(got) - sp.sympify(expected)) == 0
    results.append((label, "PASS" if ok else "FAIL"))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")


def note_unverifiable(label, reason):
    results.append((label, "SKIP"))
    print(f"[SKIP] {label} — {reason}")


# ===== 記号 =====
t = sp.symbols('t', nonnegative=True)
mA, mB, k, L, v0 = sp.symbols('m_A m_B k L v_0', positive=True)
m, M, V, phi = sp.symbols('m M V phi', positive=True)
M_tot = mA + mB
omega = sp.sqrt(k * (mA + mB) / (mA * mB))   # ω = sqrt(k(1/mA+1/mB))

# ===== 〔I〕 =====
# (2) s = x_B - x_A, r = s - L は r'' = -ω^2 r （単振動）
# (3) 初期条件 r(0)=0, r'(0)=ẋ_B-ẋ_A = 0 - v0 = -v0
r = -(v0 / omega) * sp.sin(omega * t)
s = L + r
# r'' = -ω^2 r を満たすか
check("I-(2) r''=-ω^2 r", sp.diff(r, t, 2), -omega**2 * r)
check("I-(3) r(0)=0", r.subs(t, 0), 0)
check("I-(3) r'(0)=-v0", sp.diff(r, t).subs(t, 0), -v0)

# (4) 重心運動 + 相対運動の合成
XG0 = (mA * (-L/2) + mB * (L/2)) / M_tot
VG = mA * v0 / M_tot
XG = XG0 + VG * t
xA = XG - (mB / M_tot) * s
xB = XG + (mA / M_tot) * s
# 整理形（解答に書く形）と一致するか
xA_ans = -L/2 + (mA * v0 / M_tot) * t + (mB * v0 / (M_tot * omega)) * sp.sin(omega * t)
xB_ans = L/2 + (mA * v0 / M_tot) * t - (mA * v0 / (M_tot * omega)) * sp.sin(omega * t)
check("I-(4) x_A 整理形", xA, xA_ans)
check("I-(4) x_B 整理形", xB, xB_ans)
# 初期条件・整合性
check("I-(4) x_A(0)=-L/2", xA.subs(t, 0), -L/2)
check("I-(4) x_B(0)=+L/2", xB.subs(t, 0), L/2)
check("I-(4) x_B-x_A = s", sp.simplify(xB - xA), s)
check("I-(4) ẋ_A(0)=v0", sp.diff(xA, t).subs(t, 0), v0)
check("I-(4) ẋ_B(0)=0", sp.diff(xB, t).subs(t, 0), 0)
# 運動量保存 mA ẋ_A + mB ẋ_B = mA v0 （一定）
check("I 運動量保存", sp.simplify(mA*sp.diff(xA, t) + mB*sp.diff(xB, t)), mA*v0)

# ===== 〔II〕 =====
# (2) 正面弾性衝突 φ=0: u = 2mV/(M+m)
u2 = sp.symbols('u2', positive=True)
# 1D 弾性: mV = m v' + M u, (1/2)mV^2=(1/2)m v'^2+(1/2)M u^2
vp = sp.symbols('vp')
sol = sp.solve([m*V - m*vp - M*u2, m*V**2 - m*vp**2 - M*u2**2], [vp, u2], dict=True)
# u2 != 0 の解を拾う
u_nontrivial = [d[u2] for d in sol if sp.simplify(d[u2]) != 0]
check("II-(2) u=2mV/(M+m)", u_nontrivial[0], 2*m*V/(M+m))

# (3) M=m: u = V cosφ
# 運動量: V = v cosθ + u cosφ ; 0 = v sinθ - u sinφ ; エネルギー: V^2 = v^2 + u^2
u3, v3, th = sp.symbols('u3 v3 theta', positive=True)
eqs = [V - v3*sp.cos(th) - u3*sp.cos(phi),
       v3*sp.sin(th) - u3*sp.sin(phi),
       V**2 - v3**2 - u3**2]
# θ, v を消去して u を求める：θ消去で v3^2 = V^2 - 2 V u3 cosφ + u3^2
v3sq = V**2 - 2*V*u3*sp.cos(phi) + u3**2
# エネルギーより v3^2 = V^2 - u3^2 と等置
u_sol = sp.solve(sp.Eq(v3sq, V**2 - u3**2), u3)
u_nz = [r for r in u_sol if sp.simplify(r) != 0]
check("II-(3) u=V cosφ", u_nz[0], V*sp.cos(phi))
# 陽子に与えるエネルギー (1/2)m u^2 の最大 (φ=0) は (1/2)m V^2
E = sp.Rational(1, 2)*m*(V*sp.cos(phi))**2
check("II-(3) E_max=(1/2)mV^2", E.subs(phi, 0), sp.Rational(1, 2)*m*V**2)

# モデル化・図の読み取りは数値検算対象外
note_unverifiable("I-(1) 運動方程式の符号（自由体図）", "ばね伸びの向き・力の符号は作図判断で数値検算外")

# ===== サマリ =====
fails = [l for l, s in results if s == "FAIL"]
print(f"\n--- {len(results)} 件: "
      f"PASS {sum(s=='PASS' for _, s in results)}, "
      f"FAIL {len(fails)}, "
      f"SKIP {sum(s=='SKIP' for _, s in results)} ---")
if fails:
    raise SystemExit(1)
