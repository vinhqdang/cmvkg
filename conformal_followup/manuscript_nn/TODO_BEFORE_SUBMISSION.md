# Before submitting to Neural Networks

- Competing-interest and funding statements are filled in (none declared, no specific funding).
- Elsevier policy requires a statement on the use of generative AI tools in manuscript preparation, if any were used;
  add it (title-page section "Declaration of generative AI and AI-assisted technologies") according to your own practice.
- Re-run `make_tables.py` / `make_figures.py` in `../revision` after the extended POPE and AMBER data are added,
  then copy `tables/` and the `fig_*.png` files here and rebuild (`latexmk -pdf main.tex`).
- Cover letter: not yet written (needs the final numbers).
- Neural Networks uses single-anonymised review, so the author block is real. If the editor asks for an anonymised
  copy, remove the `\author`/`\address` block.
