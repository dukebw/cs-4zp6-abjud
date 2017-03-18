SRS=SoftwareRequirementsSpecification
POC=ProofOfConceptPlan
TEST=TestPlan
DESIGNDOC=DesignDoc
SYSARCHITECTURE=SystemArchitecture
USERMANUAL=UserManual

.phony: srs poc test_plan user_manual

all: srs poc test_plan user_manual

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

test_plan:
	xelatex $(TEST).tex
	xelatex $(TEST).tex
	$(MAKE) clean
	mv $(TEST).pdf ..

user_manual:
	xelatex $(USERMANUAL).tex
	xelatex $(USERMANUAL).tex
	$(MAKE) clean
	mv $(USERMANUAL).pdf ..

design_doc:
	xelatex $(DESIGNDOC).tex
	xelatex $(DESIGNDOC).tex
	$(MAKE) clean
	mv $(DESIGNDOC).pdf ..
	xelatex $(SYSARCHITECTURE).tex
	xelatex $(SYSARCHITECTURE).tex
	$(MAKE) clean
	mv $(SYSARCHITECTURE).pdf ..

clean:
	rm -rf *.acn *.alg *.glg *.glo *.gls *.glsdefs *.ist *.xdy *.acr *.bbl *.blg *.aux *.dvi *.log *.lot *.idx *.toc *.lof *.brf *.out

mrproper: clean
	rm -rf *.pdf
