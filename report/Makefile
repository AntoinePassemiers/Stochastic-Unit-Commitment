TEXS=$(wildcard chapters/*.tex)

all: report.pdf

report.pdf: report.tex ${TEXS}
	xelatex $<
	xelatex $<

clean:
	rm -rf *.default *.toc *.log *.lod *.toc *.aux *.out */*.aux */*.default
