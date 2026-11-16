#!/bin/bash
# Build report.pdf.  Any \nXxx number macro used in the text but not produced by
# make_figures.py is defined as a red "??" (listed on stdout) so it can't go unnoticed.
cd "$(dirname "$0")"
python3 - <<'PY'
import re, glob
used = set()
for f in ["report.tex"] + glob.glob("sections/*.tex"):
    used |= set(re.findall(r"\\(n[A-Z][A-Za-z]*)", open(f).read()))
have = set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", open("numbers.tex").read()))
miss = sorted(used - have)
open("missing.tex", "w").write("".join(f"\\providecommand{{\\{m}}}{{\\textcolor{{red}}{{??}}}}\n" for m in miss))
print("MISSING NUMBERS:", ", ".join(miss) if miss else "none")
PY
[ -f sections/scale_eps_result.tex ] || echo '\textcolor{red}{[scale-aware eps result pending]}' > sections/scale_eps_result.tex
pdflatex -interaction=nonstopmode -halt-on-error report > build.log 2>&1 && bibtex report >> build.log 2>&1
pdflatex -interaction=nonstopmode report >> build.log 2>&1; pdflatex -interaction=nonstopmode report >> build.log 2>&1
grep -E "^!|Warning.*undefined" report.log | head -20; grep -o "missing: [^ ]*" report.log | sort -u
ls -la report.pdf
