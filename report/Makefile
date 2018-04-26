TEXS=$(wildcard chapters/*.tex)

all: report.pdf

report.pdf: report.tex ${TEXS}
	xelatex $< # Find cite commands and put them in a .aux file
	bibtex report  # Create a .bbl file containing the bibliography
	xelatex $< # Keep track of the references
	xelatex $< # Resolve the references

clean:
	rm -rf *.default *.bbl *.blg *.toc *.log *.lod *.toc *.aux *.out
