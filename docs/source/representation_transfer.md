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
