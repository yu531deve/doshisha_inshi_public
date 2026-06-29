#!/usr/bin/env bash
# LaTeX ビルド補助 — xelatex を2回回し,PDF を隣の pdf/ に置いて中間物を消す。
#
# 想定レイアウト:
#   <年度>/tex/<name>.tex   ← ソース
#   <年度>/<name>.pdf       ← 出力（年度直下。自動生成）
#
# 使い方（tex を引数に。相対/絶対どちらでも可）:
#   ../../../00_common/build.sh tex/doshisha_bme_2025_math.tex
#   00_common/build.sh 02_solutions/2025/tex/doshisha_bme_2025_math.tex
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: build.sh <tex/file.tex>" >&2
  exit 1
fi

tex="$1"
texdir="$(cd "$(dirname "$tex")" && pwd)"
base="$(basename "$tex" .tex)"

# 出力先: tex/ の親（年度直下）。tex/ 以外なら同じ場所。
if [ "$(basename "$texdir")" = "tex" ]; then
  outdir="$(dirname "$texdir")"
else
  outdir="$texdir"
fi
mkdir -p "$outdir"

cd "$texdir"
xelatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error "$base.tex" >/dev/null

# PDF を pdf/ へ移動,中間物は掃除（tex は残す）
mv -f "$base.pdf" "$outdir/$base.pdf"
rm -f "$base".{aux,log,out,toc,fls,fdb_latexmk,synctex.gz,nav,snm} 2>/dev/null || true

echo "OK: $outdir/$base.pdf"
