# Build the lecture notes:  make        (needs TeX Live with LuaLaTeX, latexmk)
#                           make data   (regenerate figure data; needs numpy + scipy)
#                           make clean
MAIN = main
PDF  = $(MAIN).pdf
FINAL = MAT1510_Mathematics_for_AI_Safety_Notes_L1-8.pdf

all: $(FINAL)

$(PDF): $(MAIN).tex preamble.tex frontmatter.tex backmatter.tex references.tex $(wildcard lectures/*.tex) $(wildcard figures/data/*.dat)
	latexmk -lualatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

$(FINAL): $(PDF)
	cp $(PDF) $(FINAL)

data:
	python3 figures/gen_data.py

clean:
	latexmk -C $(MAIN).tex
	rm -f $(FINAL)

.PHONY: all data clean
