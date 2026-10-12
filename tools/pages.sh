#!/bin/zsh
# Render pages of a chapter test PDF to PNG so you can LOOK at them (Read tool on the PNG files).
#   usage: tools/pages.sh N FIRST LAST [DPI]      -> $OUTROOT/LN/pg-<page>.png
path+=(/opt/anaconda3/bin)   # poppler tools (pdftoppm); harmless if the directory does not exist
N=$1; A=$2; B=$3; DPI=${4:-70}
OUTROOT=${OUTROOT:-${TMPDIR:-/tmp}/mat1510-chapters}
D=$OUTROOT/L$N
pdftoppm -r "$DPI" -f "$A" -l "$B" -png "$D/test.pdf" "$D/pg"
ls "$D"/pg-*.png
