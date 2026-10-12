#!/bin/zsh
# Build a standalone test PDF for ONE lecture chapter (so that several authors can work in parallel).
#
#   usage:  tools/build_chapter.sh N          N = 5, 6, 7 or 8   (also works for 1-4)
#
# The test document = preamble + chapter N (\input{lectures/lecture0N}) + a bibliography made of
# references.tex plus the '%%%% REFS' section of fragments/L0N.tex.  Labels of Lectures 1-4 are preloaded
# from tools/prev_labels.tex, so \ref to them resolve (page numbers there are meaningless).
# Output directory:  $OUTROOT/LN/   (test.pdf, test.log, page PNGs if you run tools/pages.sh)
set -e
path+=(/opt/anaconda3/bin)   # poppler tools (pdfinfo, pdftoppm); harmless if the directory does not exist
PY=${PYTHON:-python3}
N=$1
[[ -z "$N" ]] && { echo "usage: tools/build_chapter.sh N"; exit 1; }
ROOT=${ROOT:-${0:A:h:h}}            # project root = parent of tools/
OUTROOT=${OUTROOT:-${TMPDIR:-/tmp}/mat1510-chapters}
D=$OUTROOT/L$N
mkdir -p "$D"
cd "$ROOT"
$PY -I tools/mkbib.py "$N" "$D/bib.tex" > /dev/null
cat > "$D/test.tex" <<EOF
% !TEX program = lualatex
\documentclass[a4paper, fontsize=10.5pt, oneside, openany, headings=big]{scrbook}
\input{preamble}
\AtBeginDocument{\makeatletter\input{tools/prev_labels.tex}\makeatother}
\begin{document}
\mainmatter
\setcounter{chapter}{$((N-1))}
\input{lectures/lecture0$N}
\backmatter
\input{$D/bib}
\end{document}
EOF
latexmk -lualatex -interaction=nonstopmode -outdir="$D" "$D/test.tex" > "$D/latexmk.out" 2>&1 || true
$PY -I tools/check_log.py "$D/test.log"
pdfinfo "$D/test.pdf" 2>/dev/null | grep Pages || echo "NO PDF PRODUCED (see $D/test.log)"
echo "PDF: $D/test.pdf"
