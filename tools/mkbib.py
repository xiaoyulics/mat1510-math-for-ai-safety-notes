#!/usr/bin/env python3
"""Assemble a test bibliography: references.tex + the '%%%% REFS' section of fragments/L0N.tex.

usage (from the project root):  python3 -I tools/mkbib.py N OUTFILE
"""
import re
import sys

n, out = int(sys.argv[1]), sys.argv[2]
ref = open("references.tex", encoding="utf-8").read()
extra = ""
try:
    txt = open(f"fragments/L0{n}.tex", encoding="utf-8").read()
    m = re.search(r"^%%%% REFS[^\n]*\n(.*?)(?=^%%%% |\Z)", txt, re.S | re.M)
    if m:
        extra = m.group(1)
except FileNotFoundError:
    pass
have = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", ref))
keep = []
for entry in re.split(r"(?=\\bibitem)", extra):
    mm = re.match(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", entry)
    if mm and mm.group(1) in have:
        continue  # already in references.tex
    keep.append(entry)
ref = ref.replace("\\end{thebibliography}", "".join(keep) + "\n\\end{thebibliography}")
open(out, "w", encoding="utf-8").write(ref)
print("bibliography written:", out, f"({len(keep)} fragment entries added)")
