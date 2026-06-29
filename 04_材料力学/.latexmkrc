# XeLaTeX 固定。過去問転記 tex の中間物は build/ へ逃がす。
# （最終の解答 PDF は 02_solutions/<年度>/ 内で xelatex を直接2回回し、tex の隣に置く）
$pdf_mode = 5;        # 5 = xelatex
$xelatex  = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$out_dir  = 'build';
$aux_dir  = 'build';
