SRS=SoftwareRequirementsSpecification
POC=ProofOfConceptPlan

.phony: srs poc

all: srs poc

srs:
	xelatex $(SRS).tex
	bibtex $(SRS).aux
	xelatex $(SRS).tex
	xelatex $(SRS).tex
	$(MAKE) clean
	mv $(SRS).pdf ..

poc:
	xelatex $(POC).tex
	xelatex $(POC).tex
	$(MAKE) clean
	mv $(POC).pdf ..

clean:
	rm -rf *.acn *.alg *.glg *.glo *.gls *.glsdefs *.ist *.xdy *.acr *.bbl *.blg *.aux *.dvi *.log *.lot *.idx *.toc *.lof *.brf *.out

mrproper: clean
	rm -rf *.pdf
