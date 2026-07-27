# Implementation Plan

## Phase 1: Standalone Prototype

Build the Enhanced Five-Block MNIST topology as an editable graph workbench.

Deliverables:

- graph schema for nodes, edges, tensors, and project metadata;
- live shape inference for conv, residual blocks, pooling, feature heads, and classifiers;
- validation for MNIST output rules and multi-branch channel splits;
- inspector-driven parameter editing;
- shape trace, validation, IR, and experiment manifest panels.

Completion criteria:

- The app opens with the Enhanced Five-Block network.
- Selecting a node shows its parameters and computed output shape.
- Changing Block 5 width updates pooling and classifier dependencies.
- Setting a block width to a value not divisible by four shows a validation error.

## Phase 2: Persistence and History

Deliverables:

- project JSON export/import;
- undo and redo for parameter edits;
- named snapshots;
- deterministic topology version IDs.

Completion criteria:

- A topology can be edited, saved, reloaded, and restored without losing computed state.

Current status:

- Basic JSON export/import is implemented.
- Undo/redo is implemented for parameter, layout, node, and edge edits.
- Canvas nodes can be dragged and saved through JSON export/import.
- Residual blocks, auxiliary heads, and feature heads can be added from the library panel.
- Selected nodes can be deleted, with simple main-path rewiring where possible.
- Graph edges can be added through link mode and removed from the inspector.
- Named snapshots and deterministic topology version IDs remain open.

## Phase 3: Compiler

Deliverables:

- stable intermediate representation;
- generated PyTorch modules;
- generated smoke tests;
- Markdown and Mermaid architecture reports.

Completion criteria:

- A valid graph generates runnable PyTorch and passes a forward/backward smoke test.

Current status:

- Stable `architecture.json` IR is generated from the editable graph.
- `blocks.py`, `model.py`, `smoke_test.py`, `architecture.md`, and `experiment_manifest.yaml` previews are generated in the UI.
- A downloadable JSON bundle contains all generated files.
- `npm run generate` writes generated files to `generated/`.
- `npm run verify:generated` regenerates files and runs the emitted PyTorch smoke test.
- The default generated PyTorch model imports and passes forward/backward smoke verification.

## Phase 4: Experiment Factory

Deliverables:

- property sweeps;
- ablation recipes;
- seed expansion;
- stable experiment IDs;
- YAML and JSON manifests.

Completion criteria:

- A 4-width x 3-pooling x 3-seed sweep creates 36 unique run configs.

## Phase 5: OOF Integration

Deliverables:

- five-fold run manifests;
- OOF logits, probabilities, embeddings, labels, and indices;
- result overlays on topology versions;
- ensemble contribution scoring.

Completion criteria:

- Architectures can be ranked by validation accuracy, calibration, unique corrections, and OOF ensemble contribution.

## Phase 4.5: Generated Model Training Check

Deliverables:

- generated-model MNIST training script;
- `npm run train:generated`;
- metrics JSON export;
- local checkpoint export;
- generated-run metrics visible in the editor UI.

Current status:

- `scripts/train_generated.py` trains `generated/model.py` on a small real MNIST subset.
- `npm run train:generated` regenerates files, trains one CPU epoch on 1,024 train samples, evaluates 512 test samples, saves `generated/train_metrics.json`, and writes a local checkpoint.
- The UI reads the latest metrics through `/api/generated-metrics`.
- Latest smoke run passed: loss decreased from 2.546 to 2.140 and tiny-subset test accuracy reached 14.3%.
