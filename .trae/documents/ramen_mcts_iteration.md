# Ramen MCTS iteration workspace

- Upstream repository: `xulai1001/umaai-rs`
- Upstream branch: `ramen_workbench`
- Locked starting SHA: `a6e1f481ee4c6367b8ae1f876721a8889c4ae586`
- Fork repository: `xf8410/umaai-rs`
- Fork baseline branch: `ramen_mcts_baseline`
- Iteration branch: `workbench/ramen-mcts-iteration`

## Workflow

All implementation and GitHub Actions experiments for this iteration run only in `xf8410/umaai-rs`.
Subsequent commits are pushed to `workbench/ramen-mcts-iteration` and automatically appear in its existing fork-internal pull request targeting `ramen_mcts_baseline`.
Do not dispatch iterative CI in `xulai1001/umaai-rs`.
An upstream pull request, if ever needed, is a separate final promotion step after validation and explicit approval.

## Intended policy constraints

1. Eating ramen must be followed by training.
2. Before eating, raise the prior for a secondary-stat ramen plus its covered training; the uplift depends on the number of wisdom cards.
3. After eating, add weight to training positions covered by the selected ramen.
4. Other decisions are intended for Monte Carlo evaluation rather than additional hard gates.
