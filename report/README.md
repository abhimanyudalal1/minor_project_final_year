# Minor project report

Open **[FINAL_REPORT.docx](FINAL_REPORT.docx)** for the Word edition, or **[FINAL_REPORT.md](FINAL_REPORT.md)** for the editable Markdown source. Both follow the sample report's five-chapter academic structure and use the project's completed experiments.

- `FINAL_REPORT.md`: editable full report, including front matter, equations, results, discussion, references, and appendices.
- `FINAL_REPORT.docx`: Word edition with editable equations and tables, embedded plots and diagrams, page-number fields, and highlighted pending-result placeholders.
- `export_docx.py`: rebuilds the Word edition using `python-docx`, `markdown-it-py`, `mdit-py-plugins`, `latex2mathml`, and `mathml2omml`; no external document converter is needed.
- `figures/`: generated plots in PNG and PDF formats. The main report embeds PNGs.
- `COMPLETION_CHECKLIST.md`: administrative fields, pending experiments, and remaining figure/layout work.
- `build_figures.py`: reproducible CPU-only plotting and table assembly from existing results.
- `evidence/`: exact table values, full summary CSV, source hashes, and drafting-time status snapshots.

Open the Markdown in a viewer supporting mathematical notation and Mermaid. Numerical figures are ordinary linked image files. Keep the report folder intact when moving the draft so image links continue to work. Links to original project artifacts require the surrounding repository.

The report's scientific scope currently comprises the custom 2,000-trajectory study, custom 200-trajectory seed-0 study, and PDEBench 200-trajectory pilot. Constrained U-Net and PDEBench 600 results have explicit placeholders. The draft does not invent their outcomes.
