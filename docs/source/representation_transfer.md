# HyPER frozen representation transfer

The representation study operates only on already-trained classification-only,
reconstruction-only, and joint checkpoints. It never updates their parameters.

The exact event representation is assembled in
`HyPER.models.classification.pool_event_representation`: node, edge, and
hyperedge embeddings are each pooled by event with their existing mean and max,
then concatenated with the final global event embedding. This tensor is both
`final_event` and `classification_head_input`; it is passed unchanged to
`Classification.mlp_class`. The native event probability is `sigmoid(logit)`.
`block_0`, `block_1`, and `block_2` are the naturally produced global event
embeddings after the corresponding message-passing blocks.

The strict zero-shot ttbar single-lepton score is the fixed product of the four
selected reconstruction-role probabilities. Its definition lives in
`HyPER.topology.reconstruction_score` and is shared with the existing
reconstruction plots. No class label is used to construct or tune it.

`tools/export_hyper_representations.py` exports deterministic validation or test
subsets. `tools/compute_hyper_cka.py` computes linear centred CKA with float64
feature products and no event-by-event kernel. `tools/fit_hyper_procrustes.py`
fits the paired label-free orthogonal map on validation events only.
`tools/evaluate_hyper_head_transfer.py` streams the test split through frozen
models and evaluates direct, aligned, shuffled-pair, and random-orthogonal
controls. Every cross-model operation joins by `source_event_index` and checks
truth labels and fully-matched flags for agreement.

The end-to-end command is `tools/run_hyper_representation_transfer_study.py
--help`. It requires explicit configs, checkpoints, run directories, source H5,
split cache, output directory, event counts, and seed. The run directories are
used only to resolve the exact graph DB and existing reconstruction prediction
parts recorded in their run manifests; ambiguous filesystem searches are not
performed.

The Procrustes fit may access only paired source and target representations and
their event indices. Classification labels, reconstruction truth, AUC, and test
performance are not passed into fitting. A successful transfer therefore means
that paired label-free alignment made the reconstruction representation
functionally compatible with a frozen classification head; it does not prove
the representations contain identical information.

## Final production analysis

The production workflow is intentionally split into fitting, descriptive
representation comparison, held-out evaluation, and statistical uncertainty.
These operations use different event samples:

- `100000` validation events fit each paired Procrustes map and each of the 50
  shuffled-correspondence controls;
- `100000` test events produce the cross-layer CKA matrices;
- the complete `912666`-event test split produces streamed `final_event` CKA,
  all frozen-model scores, ROC curves, AUCs, operating points, and transfer
  controls;
- 2000 paired stratified Poisson bootstrap replicates act only on the saved
  full-test scores and do not rerun the neural networks.

The three 128-dimensional block representations are the global event state
`u_embed` after message-passing blocks 0, 1, and 2. They are not pooled node,
edge, or hyperedge matrices. `final_event` is the 1024-dimensional concatenation
of node mean/max, edge mean/max, the final global event state, and hyperedge
mean/max. `classification_head_input` is an explicit alias of `final_event`; it
is retained in machine-readable exports for interface verification but omitted
from primary CKA heatmaps to avoid duplicated rows and columns.

`tools/compute_hyper_streaming_cka.py` calculates complete-test final-event CKA
from float64 sufficient statistics (`sum_X`, `sum_Y`, `XTX`, `YTY`, and `XTY`).
It never stores an event-by-event kernel or a complete full-test representation
matrix. `tools/fit_hyper_alignment_controls.py` fits one paired and 50 shuffled
orthogonal maps per direction while loading only event indices and the requested
representations. `tools/evaluate_hyper_transfer_production.py` evaluates the
three frozen backbones once per test batch and applies alignment controls in
small vectorised chunks, writing principal scores and control ensembles to
memory-mapped NumPy arrays.

The strict reconstruction zero-shot score remains

```text
S_reco = p_top1 * p_top2 * p_W1 * p_W2
```

and is treated as a continuous ranking score. ROC curves sweep its threshold;
no fixed cut is required to calculate AUC. The final plots include the score
split into background, fully matched signal, and non-fully-matched signal,
category-specific ROC curves, and background rejection at representative signal
efficiencies.

The shuffled-control distribution and the paired bootstrap answer different
questions. Shuffled alignments test whether correct event correspondence gives a
better map than the no-correspondence null. The bootstrap estimates finite-test-
sample uncertainty for fixed methods using the same event weights for every
score. They are reported separately.

The production submission entry point is:

```bash
scripts/submit_hyper_representation_transfer_production.sh
```

It submits explicit Slurm stages for exports, alignment controls, CKA, streamed
full-test CKA, full-test evaluation, statistical summaries, plotting, and final
validation. No source file under the original checkpoint runtime is modified.

Before the complete run, submit the same dependency graph in production-shaped
smoke mode:

```bash
scripts/submit_hyper_representation_transfer_production.sh smoke
```

This uses 2000 alignment events, 2000 CKA events, 2000 held-out test events,
three shuffled maps per direction, three random controls, and 50 paired
bootstrap replicates. The final production submission is:

```bash
scripts/submit_hyper_representation_transfer_production.sh production
```

Completed stage outputs are reused only after their event counts, array shapes,
metadata, and required plots pass explicit checks. Partial outputs fail rather
than being silently reused; setting `HYPER_OVERWRITE=1` deliberately replaces
such outputs. During full-test transfer evaluation, all paired, shuffled, and
random rotation matrices are copied to the allocated GPU once. Each frozen
backbone is then evaluated once per event batch, with the control maps applied
in vectorised chunks. The long-form `operating_points.csv` records the score
threshold, background efficiency, and background rejection at 50%, 70%, and
80% signal efficiency for every method and signal subset.
