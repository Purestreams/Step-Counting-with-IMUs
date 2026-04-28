# Presentation README

This folder contains the Beamer slide deck for the IMU step-counting project.

## Output

- Main source: `template.tex`
- Generated PDF: `template.pdf`
- Slide count: 12 pages

## Build With WSL

From PowerShell, run:

```powershell
wsl sh -lc "cd /mnt/c/Users/mio.zhu/Documents/Step-Counting-with-IMUs/presentation && pdflatex -interaction=nonstopmode -halt-on-error template.tex && pdflatex -interaction=nonstopmode -halt-on-error template.tex"
```

The command runs `pdflatex` twice so Beamer navigation metadata is refreshed.

## Slide Files

- `01-title.tex`: title page
- `03-example-motivation.tex`: problem definition and interface constraints
- `04-example-theory.tex`: dataset and ground truth
- `05-example-conclusion.tex`: preprocessing and windowing pipeline
- `06-example-appendix.tex`: Causal TCN StepNet architecture, model comparison, and ACF baseline
- `07-example-references.tex`: training objective, hyperparameters, and early-stopping behavior
- `08-example-appendix-figures.tex`: offline and real-time inference
- `09-example-appendix-terms.tex`: evaluation results and subset generalization analysis
- `10-example-appendix-code.tex`: real-time demo, validity threats, and future work

## Notes

- The deck uses the HKU Beamer theme assets under `images/beamer/`.
- The generalization slide imports the report figure from `../report/fig/calibration_subset_abs_error.png`.
- `biblatex` and Open Sans were removed from the template because the current WSL TeX Live install does not include those packages and the deck does not use citations.
- If the slides are edited, rebuild `template.pdf` with the WSL command above.