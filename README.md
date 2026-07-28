# MNIST Neural Network Topology Editor

A focused visual topology editor and compiler workbench for the MNIST research models.

The first implementation targets the Enhanced Five-Block Multi-Branch CNN:

- structured graph data as the source of truth;
- live tensor shape propagation;
- topology validation before code generation;
- parameter and FLOP estimates;
- intermediate representation and experiment manifest previews.

## Run Locally

```powershell
npm install
npm run dev
```

Then open the local URL printed by Next.js.

Generate and verify the default topology:

```powershell
npm run generate
npm run verify:generated
npm run train:generated
```

`npm run train:generated` writes the latest generated files and metrics to `generated/`, and stores each training attempt under `runs/<timestamp>_<topology-id>/`.

## Current Scope

This repository starts with the Phase 1 dashboard prototype:

- high-level graph canvas;
- template browser;
- node inspector;
- editable block and head parameters;
- draggable node layout;
- add/delete for residual blocks and heads;
- connect/disconnect graph edges;
- undo and redo for parameter edits;
- project JSON import and export;
- live shape trace;
- validation panel;
- repair cue for invalid branch channel splits;
- generated previews for `architecture.json`, `blocks.py`, `model.py`, `smoke_test.py`, `architecture.md`, and `experiment_manifest.yaml`;
- generated file bundle download.
- generated output files in `generated/`;
- runtime verification for generated PyTorch with `npm run verify:generated`.
- small real-MNIST training check with `npm run train:generated`;
- generated training metrics surfaced in the UI from `generated/train_metrics.json`;
- generated training can be launched from the editor UI with configurable sample limits;
- generated checkpoints saved locally under `generated/`.
- live generated training progress, logs, and saved run history in the editor UI;
- stable topology IDs for comparing a saved result with the current canvas.
- metric charts for accuracy, train loss, test loss, and live batch loss;
- ablation queue for running multiple topology variants one after another.
- queue presets for width, drop-path, pooling mode, and SE on/off sweeps;
- queue stop/cancel, queued-item clearing that keeps completed results, JSON/CSV comparison export, and click-to-load variants back onto the canvas.

Next phases will add OOF-aware experiment generation and richer export/reporting for completed runs.
