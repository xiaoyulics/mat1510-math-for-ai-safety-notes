# latexmk configuration: build with LuaLaTeX
$pdf_mode = 4;                       # 4 = lualatex
$lualatex = 'lualatex -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 0;                     # bibliography is hand-written (thebibliography)
