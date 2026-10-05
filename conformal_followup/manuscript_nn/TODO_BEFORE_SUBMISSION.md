# Before submitting to Neural Networks

- Competing-interest and funding statements are filled in (none declared, no specific funding).
- The generative-AI declaration is in `main.tex` (before the competing-interest statement). Elsevier's template also asks for the tool name; it is left out on the author's instruction.
- Re-run `make_tables.py` / `make_figures.py` in `../revision` after the extended POPE and AMBER data are added,
  then copy `tables/` and the `fig_*.png` files here and rebuild (`latexmk -pdf main.tex`).
- Cover letter: not yet written (needs the final numbers).
- Neural Networks uses single-anonymised review, so the author block is real. If the editor asks for an anonymised
  copy, remove the `\author`/`\address` block.
