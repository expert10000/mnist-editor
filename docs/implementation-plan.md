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
