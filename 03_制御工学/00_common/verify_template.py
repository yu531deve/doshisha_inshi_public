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


# === ここに各小問の検算を書く ============================
# 例（数学）:
#   x = sp.symbols('x')
#   check("(1) 広義積分", sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo)), sp.sqrt(sp.pi))
#
# 例（力学/材料力学）: 公式値を数値で確認
#   check("(2) 固有角振動数", mp.sqrt(k/m), 12.5)
#
# 例（制御工学）: 伝達関数の極で安定判別
#   s = sp.symbols('s'); poles = sp.solve(s**2 + 3*s + 2, s)
#   check("(3) 安定（全極の実部<0）", all(sp.re(p) < 0 for p in poles), True)
#
# 例（証明・図読み取り）:
#   note_unverifiable("(4) 自由体図", "図からのモデル化で、数値検算の対象外")
# =========================================================


if __name__ == "__main__":
    n_pass = sum(1 for _, r in results if r is True)
    n_fail = sum(1 for _, r in results if r is False)
    n_skip = sum(1 for _, r in results if r is None)
    print(f"\n--- {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP ---")
    raise SystemExit(1 if n_fail else 0)
