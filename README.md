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

## Current Scope

This repository starts with the Phase 1 dashboard prototype:

- high-level graph canvas;
- template browser;
- node inspector;
- editable block and head parameters;
- undo and redo for parameter edits;
- project JSON import and export;
- live shape trace;
- validation panel;
- repair cue for invalid branch channel splits;
- IR preview;
- experiment sweep preview.

Next phases will add save/load, undo/redo, generated PyTorch, smoke tests, ablations, and OOF-aware experiment generation.
