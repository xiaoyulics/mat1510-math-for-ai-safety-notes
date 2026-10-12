#!/usr/bin/env python3
"""Summarize a LaTeX log: errors, undefined references/citations, overfull boxes, other warnings.

usage: python3 -I tools/check_log.py path/to/file.log
"""
import re
import sys

log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
lines = log.splitlines()

errors = [l for l in lines if re.match(r"^\S+:\d+: ", l) or l.startswith("! ")]
print(f"== errors: {len(errors)}")
for e in errors[:25]:
    print("  ", e)

und_ref = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
und_cit = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", log)))
mult = sorted(set(re.findall(r"Label `([^']+)' multiply defined", log)))
print(f"== undefined references: {len(und_ref)}", und_ref[:40])
print(f"== undefined citations: {len(und_cit)}", und_cit[:40])
print(f"== multiply defined labels: {len(mult)}", mult[:20])

over = []
for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)[^\n]*", log):
    w = float(m.group(1))
    if w >= 1.0:
        over.append((w, m.group(0)[:110]))
print(f"== overfull hboxes >= 1pt: {len(over)}")
for w, t in sorted(over, reverse=True)[:15]:
    print(f"   {w:6.1f}pt  {t}")

under = len(re.findall(r"Underfull \\vbox", log))
print(f"== underfull vbox: {under}")

other = set()
for m in re.finditer(r"(LaTeX|Package \w+|pgfplots) Warning: ([^\n]{0,140})", log):
    t = m.group(2)
    if "Reference" in t or "Citation" in t or "multiply defined" in t or "Label(s) may have changed" in t:
        continue
    other.add((m.group(1) + ": " + t).strip())
print(f"== other warnings: {len(other)}")
for t in sorted(other)[:12]:
    print("  ", t)
