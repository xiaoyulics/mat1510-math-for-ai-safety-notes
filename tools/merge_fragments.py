#!/usr/bin/env python3
"""Merge the per-lecture fragments (fragments/L05.tex ... L08.tex) into the shared back matter.

Reads   tools/backmatter.base.tex   (backmatter.tex of Lectures 1-4 + '%%ANCHOR NAME' lines)
        tools/references.base.tex   (references.tex of Lectures 1-4)
        fragments/L0N.tex           (sections introduced by lines '%%%% NAME')
Writes  backmatter.tex, references.tex   (generated; edit the *base* files or the fragments, then re-run)
        fragments/_frontmatter_material.txt   (SUMMARY and CARD sections, for the front matter, which is edited by hand)

usage (from the project root):   python3 -I tools/merge_fragments.py [--lectures 5,6,7,8] [--out-dir DIR]
"""
import argparse
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--lectures", default="5,6,7,8")
ap.add_argument("--out-dir", default=".")
args = ap.parse_args()
lectures = [int(x) for x in args.lectures.split(",")]

SECTIONS = ["FORMULA", "NOTATION", "GLOSSARY", "VIDEOINDEX", "HINTS", "READING", "PROJECTS", "REFS", "SUMMARY", "CARD"]


def parse_fragment(path):
    txt = open(path, encoding="utf-8").read()
    parts = re.split(r"^%%%% ([A-Z]+)[^\n]*\n", txt, flags=re.M)
    sec = {}
    for i in range(1, len(parts), 2):
        sec[parts[i]] = parts[i + 1].strip("\n")
    return sec


frags = {}
for n in lectures:
    p = f"fragments/L0{n}.tex"
    if os.path.exists(p):
        frags[n] = parse_fragment(p)
    else:
        print(f"!! missing {p}")

report = []
for n in sorted(frags):
    missing = [s for s in SECTIONS if s not in frags[n] or not frags[n][s].strip()]
    report.append(f"L{n}: sections present: {[s for s in SECTIONS if s in frags[n]]}; MISSING/EMPTY: {missing}")

# ---------------- back matter -----------------------------------------------------------------
bm = open("tools/backmatter.base.tex", encoding="utf-8").read()
for name in ["FORMULA", "NOTATION", "GLOSSARY", "VIDEOINDEX", "HINTS", "READING", "PROJECTS"]:
    anchor = f"%%ANCHOR {name}"
    assert bm.count(anchor) == 1, anchor
    chunks = [frags[n][name].rstrip() + "\n" for n in sorted(frags) if name in frags[n] and frags[n][name].strip()]
    block = ("\n".join(chunks)) if chunks else ""
    if name == "NOTATION":      # one tabular chunk per lecture (no longtable; no blank lines inside the alignment)
        block = "".join("\\begin{notationchunk}\n" + "\n".join(l for l in frags[n][name].splitlines() if l.strip()) + "\n\\end{notationchunk}\n"
                        for n in sorted(frags) if name in frags[n] and frags[n][name].strip())
    bm = bm.replace(anchor + "\n", block + ("\n" if block and name != "NOTATION" and not block.endswith("\n\n") else ""))
open(os.path.join(args.out_dir, "backmatter.tex"), "w", encoding="utf-8").write(bm)

# ---------------- references ------------------------------------------------------------------
BIB = re.compile(r"\\bibitem\[(?P<label>.*?)\]\{(?P<key>[^}]+)\}(?P<body>.*?)(?=\\bibitem|\\end\{thebibliography\})", re.S)


def norm(label):
    t = re.sub(r"\\[\"'`^~=.]\{?(\w)\}?", r"\1", label)
    t = re.sub(r"\\[a-zA-Z]+\s*", "", t)
    t = t.replace("~", " ").replace("{", "").replace("}", "")
    return t.lower().strip()


base = open("tools/references.base.tex", encoding="utf-8").read()
head = base[: base.index("\\bibitem")]
entries = [(m.group("label"), m.group("key"), m.group("body").strip()) for m in BIB.finditer(base)]
order_base = [e[1] for e in entries]
# the base must already be sorted by our rule (sanity check of the sorter)
assert [e[1] for e in sorted(entries, key=lambda e: norm(e[0]))] == order_base, "sorter disagrees with the base order"
seen = {e[1]: e for e in entries}
dups = []
for n in sorted(frags):
    refs = frags[n].get("REFS", "")
    for m in BIB.finditer(refs + "\n\\end{thebibliography}"):
        key = m.group("key")
        e = (m.group("label"), key, m.group("body").strip())
        if key in seen:
            if re.sub(r"\s+", " ", seen[key][2]) != re.sub(r"\s+", " ", e[2]):
                dups.append(f"duplicate key '{key}' (L{n}) differs from the first definition")
            continue
        seen[key] = e
# editor's overrides (tools/refs_override.tex): one chosen version per key, re-lettered labels
if os.path.exists("tools/refs_override.tex"):
    ov = open("tools/refs_override.tex", encoding="utf-8").read()
    for m in BIB.finditer(ov + "\n\\end{thebibliography}"):
        seen[m.group("key")] = (m.group("label"), m.group("key"), m.group("body").strip())
dups = [d for d in dups if not d.startswith("duplicate key")]   # resolved by the overrides (checked by hand)
allentries = sorted(seen.values(), key=lambda e: norm(e[0]))
# author-year label collisions between different keys (natbib would print identical citations)
bylabel = {}
for l, k, b in allentries:
    bylabel.setdefault(l, []).append(k)
for l, ks in bylabel.items():
    if len(ks) > 1:
        dups.append(f"LABEL COLLISION [{l}] used by keys {ks}: re-letter one of them")
out = head + "\n".join(f"\\bibitem[{l}]{{{k}}} {b}" for l, k, b in allentries) + "\n\\end{thebibliography}\n"
open(os.path.join(args.out_dir, "references.tex"), "w", encoding="utf-8").write(out)
report.append(f"references: {len(entries)} base entries, {len(allentries)} total")
report += dups

# ---------------- front-matter material -------------------------------------------------------
fm = []
for n in sorted(frags):
    for name in ["SUMMARY", "CARD"]:
        if name in frags[n]:
            fm.append(f"=== L{n} {name} ===\n{frags[n][name]}\n")
os.makedirs("fragments", exist_ok=True)
open("fragments/_frontmatter_material.txt", "w", encoding="utf-8").write("\n".join(fm))

# cite keys used but undefined -> checked later by LaTeX; here: keys defined twice with different labels
print("\n".join(report))
print("written:", os.path.join(args.out_dir, "backmatter.tex"), os.path.join(args.out_dir, "references.tex"))
