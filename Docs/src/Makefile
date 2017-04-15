PROBSTMT=ProblemStatement
SRS=SoftwareRequirementsSpecification
POC=ProofOfConceptPlan
TEST=TestPlan
DETAILEDDESIGN=DetailedDesign
TESTREPORT=TestReport
SYSARCHITECTURE=SystemArchitecture
USERMANUAL=UserManual

.phony: srs poc test_plan user_manual

all: prob_stmt srs poc test_plan test_report user_manual design_doc

prob_stmt:
	xelatex $(PROBSTMT).tex
	xelatex $(PROBSTMT).tex
	$(MAKE) clean
	mv $(PROBSTMT).pdf ..


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

test_report:
	xelatex $(TESTREPORT).tex
	xelatex $(TESTREPORT).tex
	$(MAKE) clean
	mv $(TESTREPORT).pdf ..

user_manual:
	xelatex $(USERMANUAL).tex
	xelatex $(USERMANUAL).tex
	$(MAKE) clean
	mv $(USERMANUAL).pdf ..

design_doc:
	xelatex $(SRS).tex
	bibtex $(SRS).aux
	xelatex $(SRS).tex
	xelatex $(SRS).tex
	xelatex $(DETAILEDDESIGN).tex
	xelatex $(DETAILEDDESIGN).tex
	xelatex $(SYSARCHITECTURE).tex
	xelatex $(SYSARCHITECTURE).tex
	$(MAKE) clean
	mv $(SRS).pdf ..
	mv $(DETAILEDDESIGN).pdf ..
	mv $(SYSARCHITECTURE).pdf ..

clean:
	rm -rf *.acn *.alg *.glg *.glo *.gls *.glsdefs *.ist *.xdy *.acr *.bbl *.blg *.aux *.dvi *.log *.lot *.idx *.toc *.lof *.brf *.out

mrproper: clean
	rm -rf *.pdf
