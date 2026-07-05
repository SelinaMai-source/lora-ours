# LB-CL Official Code Search - 2026-07-05

Scope: Standard T5-large PEFT CL method `LB-CL` from "Learn More, but Bother Less: Parameter Efficient Continual Learning". This audit only checks official/author code availability. No training or Ours changes were launched.

## Sources Checked

- NeurIPS proceedings paper: `https://proceedings.neurips.cc/paper_files/paper/2024/file/b0bc711f48724237b38823c4d9cee10b-Paper-Conference.pdf`
- NeurIPS abstract page: `https://proceedings.neurips.cc/paper_files/paper/2024/hash/b0bc711f48724237b38823c4d9cee10b-Abstract-Conference.html`
- NeurIPS virtual poster: `https://neurips.cc/virtual/2024/poster/94599`
- NeurIPS slides: `https://neurips.cc/media/neurips-2024/Slides/94599.pdf`
- OpenReview forum ID: `ZxtaNh5UYB`
- ML Anthology page: `https://mlanthology.org/neurips/2024/qiao2024neurips-learn/`
- Author/profile searches for Fuli Qiao / `fvq5015` and Mehrdad Mahdavi.
- GitHub title/ID/hash searches for `Learn More, but Bother Less`, `LB-CL`, `ZxtaNh5UYB`, and `b0bc711f48724237b38823c4d9cee10b`.

## Evidence

- The paper reports LB-CL T5-large Standard CL order1/2/3 `76.9/76.5/76.8`, avg `76.7`.
- The NeurIPS paper checklist item 5, "Open access to data and code", answers `[No]` with the justification: "We use open-source datasets and models, but do not attach the code."
- Appendix A.2 provides implementation details such as DeepSpeed, four NVIDIA A6000 GPUs, learning rate, batch size, dropout, and regularization, but no repository URL, artifact URL, commit, license, or command line.
- The NeurIPS virtual page exposes Paper, Slides, Poster, and OpenReview links only; no Code/GitHub/project link was found.
- The slides describe the method and results only; no repository URL or code availability statement was found.
- The ML Anthology page provides citation/DOI metadata only; no code field was found.
- OpenReview browser fetch is blocked by browser verification in this environment; API attempts returned `403`. Web-indexed OpenReview snippets list authors/status/license for the paper but no code link.
- `https://github.com/mehrdadmahdavi` exists but shows only `github_tutorial`; no LB-CL repository.
- Searches for Fuli Qiao / `fvq5015` found scholar/LinkedIn style profiles and unrelated similarly named GitHub accounts, but no public author GitHub repository tied to LB-CL.

## Non-Accepted Candidate

- `https://github.com/yaoyz96/low-rank-cl` appeared in broad web search snippets.
- It is not accepted as LB-CL official code: the repository identifies an ICLR 2026 paper, "Revisiting Weight Regularization for Low-Rank Continual Learning", and methods such as InfLoRA, SD-LoRA, CL-LoRA, and EWC-LoRA.
- The repository owner/authors do not match Fuli Qiao or Mehrdad Mahdavi, and the repository is not linked from the LB-CL paper, OpenReview indexed content, NeurIPS poster/slides, ML Anthology page, or author profiles.

## Decision

- Official/author repo: not found.
- Commit: none.
- License/availability: paper is public; official code is not attached and no code license is available.
- Runnable status: not runnable; no official/author code or supplement artifact was confirmed.
- Gate: keep `LB-CL` as `paper_only_baseline`.
- Blocker: code-level reproducibility is blocked until the authors publish or explicitly identify an official repository/supplement. Do not use third-party implementations as official LB-CL code.
