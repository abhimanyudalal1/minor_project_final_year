# Report completion checklist

Main document: [FINAL_REPORT.md](FINAL_REPORT.md).

The draft contains the full five-chapter narrative, completed-study tables, equations, references, and generated plots. The items below are intentionally unresolved rather than guessed.

## Administrative information

- [ ] Replace all `[FILL: ...]` fields: university, programme, institute, department, authors, enrolment numbers, supervisor, dates, and signatures.
- [ ] Confirm institutional declaration and certification wording with the authors; the draft does not assert that anyone has signed or examined the work.
- [ ] Confirm hardware model and RAM from the experimental machine.
- [ ] Insert a public code/data URL only if a release actually exists.
- [ ] Attach the research manuscript if required, or remove Appendix A and renumber.

## Pending experiments

- [ ] Constrained U-Net, PDEBench 200: confirm both seeds and all evaluation/summary stages completed, then fill Table 4.6 and Figure 4.11.
- [ ] PDEBench 600: confirm all six runs and evaluation/summary stages completed, then fill Table 4.7 and Figure 4.12.
- [ ] Keep failure rates, unavailable aggregates, model-seed SD, and adapter labels in the new tables.
- [ ] Do not substitute a partial seed result for a two-seed mean.
- [ ] Use the same adapter, grid, seed set, and horizon for each direct comparison.
- [ ] Preserve the qualification that the extensions were selected after earlier test inspection and reuse held-out examples.
- [ ] Update the abstract, Table 3.1, completed-run count, research-question summary, discussion, and conclusion together.
- [ ] If a study is unfinished at submission, keep its protocol as future work and remove empty result tables from the submission version.

## Figures and layout

- [ ] Add Figure 3.4: reference heatmaps/profiles for the first stored custom and PDEBench test IDs; exact selection guidance is in §3.4.
- [ ] Optionally add Figure 4.10: checkpoint-based prediction/reference/error heatmaps; guidance is in §4.2. This is supplementary to the existing quantitative figures.
- [ ] Add Figures 4.11 and 4.12 only when their full evaluations exist.
- [ ] Render the two Mermaid diagrams for Word/LaTeX export if necessary.
- [ ] Use `figures/*.pdf` for vector-quality document insertion, or PNGs for Markdown.
- [ ] Generate final page numbers, lists of figures/tables, and a detailed table of contents in the final document editor.
- [ ] Choose consolidated or chapter-wise references according to the department template.
- [ ] Remove the draft-status box and resolved placeholders before submission.

## Evidence and regeneration

The current draft's generated numerical blocks and plots use only completed artifacts. Run from the project root:

```bash
MPLCONFIGDIR=/tmp/burgers-report-mpl .venv/bin/python report/build_figures.py
```

This rebuilds the existing three-study figures and table blocks; it does not incorporate new studies or rewrite their interpretation. See [evidence/source_manifest.json](evidence/source_manifest.json), [evidence/metrics.csv](evidence/metrics.csv), and [evidence/tables.md](evidence/tables.md) for the source record.
