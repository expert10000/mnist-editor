"use client";

import {
  BrainCircuit,
  CheckCircle2,
  Download,
  FileCode2,
  Link2,
  Plus,
  Redo2,
  RefreshCw,
  Table2,
  TriangleAlert,
  Trash2,
  Undo2,
  Unlink2,
  Upload,
  XCircle,
} from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState, type PointerEvent, type ReactNode } from "react";

import { generateFiles, type GeneratedFile } from "@/lib/codegen";
import {
  enhancedFiveBlockTopology,
  formatCompactNumber,
  formatShape,
  parseTopologyProject,
  resolveTopology,
  type NodeKind,
  type TopologyEdge,
  type TopologyNode,
  type TopologyProject,
} from "@/lib/topology";

type BottomPanel = "trace" | "validation" | "generated";
type EditorHistory = { past: TopologyProject[]; present: TopologyProject; future: TopologyProject[] };
type Notice = { tone: "good" | "bad"; text: string } | null;
type GeneratedTrainMetrics = {
  status: string;
  epochs: number;
  train_limit: number;
  test_limit: number;
  first_batch_loss: number;
  final_batch_loss: number;
  train_loss: number;
  test_loss: number;
  test_accuracy: number;
  checkpoint: string;
  duration_seconds: number;
  passed_smoke_rule: boolean;
} | null;
type DragState = {
  nodeId: string;
  startClientX: number;
  startClientY: number;
  startX: number;
  startY: number;
  origin: TopologyProject;
  moved: boolean;
};

export default function Home() {
  const [history, setHistory] = useState<EditorHistory>({ past: [], present: enhancedFiveBlockTopology, future: [] });
  const [selectedNodeId, setSelectedNodeId] = useState("block3");
  const [bottomPanel, setBottomPanel] = useState<BottomPanel>("trace");
  const [selectedGeneratedPath, setSelectedGeneratedPath] = useState("model.py");
  const [notice, setNotice] = useState<Notice>(null);
  const [trainMetrics, setTrainMetrics] = useState<GeneratedTrainMetrics>(null);
  const [metricsBusy, setMetricsBusy] = useState(false);
  const [connectSourceId, setConnectSourceId] = useState<string | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const suppressClickRef = useRef(false);
  const project = history.present;
  const resolution = useMemo(() => resolveTopology(project), [project]);
  const nodeById = useMemo(() => new Map(project.nodes.map((node) => [node.id, node])), [project.nodes]);
  const selectedNode = project.nodes.find((node) => node.id === selectedNodeId) ?? project.nodes[0];
  const selectedTrace = resolution.trace.find((entry) => entry.nodeId === selectedNode.id);
  const selectedEdges = project.edges.filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id);
  const generatedFiles = useMemo(() => generateFiles(project), [project]);
  const selectedGeneratedFile = generatedFiles.find((file) => file.path === selectedGeneratedPath) ?? generatedFiles[0];
  const branchRepair = getBranchRepair(selectedNode);
  const boardSize = useMemo(() => getBoardSize(project.nodes), [project.nodes]);

  async function loadGeneratedMetrics() {
    setMetricsBusy(true);
    try {
      const response = await fetch("/api/generated-metrics", { cache: "no-store" });
      const payload = await response.json();
      setTrainMetrics(payload.metrics ?? null);
    } finally {
      setMetricsBusy(false);
    }
  }

  useEffect(() => {
    void loadGeneratedMetrics();
  }, []);

  function updateNodeParameter(nodeId: string, key: string, value: boolean | number | string) {
    commitProject((current) => ({
      ...current,
      nodes: current.nodes.map((node) =>
        node.id === nodeId ? { ...node, parameters: { ...node.parameters, [key]: value } } : node,
      ),
    }));
    setNotice(null);
  }

  function updateNodePosition(nodeId: string, x: number, y: number) {
    setHistory((current) => ({
      ...current,
      present: {
        ...current.present,
        nodes: current.present.nodes.map((node) =>
          node.id === nodeId ? { ...node, position: { x: Math.max(16, x), y: Math.max(16, y) } } : node,
        ),
      },
    }));
  }

  function restoreTemplate() {
    commitProject(() => enhancedFiveBlockTopology);
    setSelectedNodeId("block3");
    setBottomPanel("trace");
    setNotice({ tone: "good", text: "Template restored." });
  }

  function commitProject(nextProject: TopologyProject | ((current: TopologyProject) => TopologyProject)) {
    setHistory((current) => ({
      past: [...current.past, current.present].slice(-50),
      present: typeof nextProject === "function" ? nextProject(current.present) : nextProject,
      future: [],
    }));
  }

  function commitCurrentDrag() {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    dragRef.current = null;
    suppressClickRef.current = drag.moved;
    if (!drag.moved) {
      return;
    }
    setHistory((current) => ({
      past: [...current.past, drag.origin].slice(-50),
      present: current.present,
      future: [],
    }));
    setNotice({ tone: "good", text: "Node position updated." });
  }

  function startNodeDrag(event: PointerEvent<HTMLButtonElement>, node: TopologyNode) {
    if (event.button !== 0) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      nodeId: node.id,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startX: node.position.x,
      startY: node.position.y,
      origin: project,
      moved: false,
    };
  }

  function moveNode(event: PointerEvent<HTMLButtonElement>) {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const dx = event.clientX - drag.startClientX;
    const dy = event.clientY - drag.startClientY;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
      drag.moved = true;
    }
    updateNodePosition(drag.nodeId, drag.startX + dx, drag.startY + dy);
  }

  function undo() {
    setHistory((current) => {
      const previous = current.past.at(-1);
      if (!previous) {
        return current;
      }
      return {
        past: current.past.slice(0, -1),
        present: previous,
        future: [current.present, ...current.future].slice(0, 50),
      };
    });
    setNotice(null);
  }

  function redo() {
    setHistory((current) => {
      const next = current.future[0];
      if (!next) {
        return current;
      }
      return {
        past: [...current.past, current.present].slice(-50),
        present: next,
        future: current.future.slice(1),
      };
    });
    setNotice(null);
  }

  function exportProject() {
    const json = JSON.stringify(project, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${project.name.toLowerCase().replaceAll(" ", "-")}.json`;
    link.click();
    URL.revokeObjectURL(url);
    setNotice({ tone: "good", text: "Project JSON exported." });
  }

  function downloadGeneratedBundle() {
    const payload = {
      generatedAt: new Date().toISOString(),
      project: project.name,
      files: Object.fromEntries(generatedFiles.map((file) => [file.path, file.content])),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${project.name.toLowerCase().replaceAll(" ", "-")}-generated-files.json`;
    link.click();
    URL.revokeObjectURL(url);
    setNotice({ tone: "good", text: "Generated file bundle downloaded." });
  }

  async function importProject(file: File | undefined) {
    if (!file) {
      return;
    }
    try {
      const parsed = JSON.parse(await file.text());
      const result = parseTopologyProject(parsed);
      if (!result.project) {
        setNotice({ tone: "bad", text: result.errors.join(" ") });
        setBottomPanel("validation");
        return;
      }
      commitProject(result.project);
      const nextSelected = result.project.nodes.some((node) => node.id === selectedNodeId) ? selectedNodeId : "block3";
      setSelectedNodeId(result.project.nodes.some((node) => node.id === nextSelected) ? nextSelected : result.project.nodes[0].id);
      setNotice({ tone: "good", text: `Imported ${result.project.name}.` });
    } catch (error) {
      setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Project JSON could not be imported." });
      setBottomPanel("validation");
    } finally {
      if (importInputRef.current) {
        importInputRef.current.value = "";
      }
    }
  }

  function repairSelectedBranchWidth() {
    if (!branchRepair) {
      return;
    }
    updateNodeParameter(selectedNode.id, "out_channels", branchRepair.repairedWidth);
    setBottomPanel("validation");
    setNotice({ tone: "good", text: `${selectedNode.name} width repaired to ${branchRepair.repairedWidth}.` });
  }

  function addNode(kind: Extract<NodeKind, "multi_branch_residual" | "auxiliary_classifier" | "feature_head">) {
    const selectedIncoming = project.edges.find((edge) => edge.target === selectedNode.id && edge.branch !== "auxiliary");
    const source =
      selectedNode.kind === "classifier" && selectedIncoming
        ? nodeById.get(selectedIncoming.source) ?? selectedNode
        : selectedNode.kind === "auxiliary_classifier"
          ? nodeById.get(project.edges.find((edge) => edge.target === selectedNode.id)?.source ?? "") ?? selectedNode
          : selectedNode;
    const id = uniqueNodeId(project, kind === "multi_branch_residual" ? "block" : kind === "auxiliary_classifier" ? "aux" : "head");
    const node = createNode(kind, id, {
      x: source.position.x + (kind === "auxiliary_classifier" ? 0 : 200),
      y: source.position.y + (kind === "auxiliary_classifier" ? 160 : 0),
    });

    if (kind === "auxiliary_classifier") {
      commitProject((current) => ({
        ...current,
        nodes: [...current.nodes, node],
        edges: [
          ...current.edges,
          { id: uniqueEdgeId(current, source.id, id), source: source.id, target: id, branch: "auxiliary" },
        ],
      }));
      setSelectedNodeId(id);
      setNotice({ tone: "good", text: "Auxiliary head added." });
      return;
    }

    commitProject((current) => {
      const outgoing = firstMainOutgoing(current, source.id);
      const edges = current.edges.filter((edge) => edge.id !== outgoing?.id);
      return {
        ...current,
        nodes: [...current.nodes, node],
        edges: [
          ...edges,
          { id: uniqueEdgeId(current, source.id, id), source: source.id, target: id },
          ...(outgoing ? [{ id: uniqueEdgeId(current, id, outgoing.target), source: id, target: outgoing.target }] : []),
        ],
      };
    });
    setSelectedNodeId(id);
    setNotice({ tone: "good", text: `${node.name} inserted into the main path.` });
  }

  function deleteSelectedNode() {
    if (selectedNode.id === "input" || selectedNode.id === "classifier") {
      setNotice({ tone: "bad", text: "Input and final classifier are required nodes." });
      return;
    }
    const incoming = project.edges.find((edge) => edge.target === selectedNode.id && edge.branch !== "auxiliary");
    const outgoing = firstMainOutgoing(project, selectedNode.id);
    commitProject((current) => {
      const edges = current.edges.filter((edge) => edge.source !== selectedNode.id && edge.target !== selectedNode.id);
      return {
        ...current,
        nodes: current.nodes.filter((node) => node.id !== selectedNode.id),
        edges:
          incoming && outgoing && selectedNode.kind !== "auxiliary_classifier"
            ? [...edges, { id: uniqueEdgeId(current, incoming.source, outgoing.target), source: incoming.source, target: outgoing.target }]
            : edges,
      };
    });
    const fallbackId = outgoing?.target ?? incoming?.source ?? "input";
    setSelectedNodeId(nodeById.has(fallbackId) ? fallbackId : "input");
    setNotice({ tone: "good", text: `${selectedNode.name} deleted.` });
  }

  function connectNodes(source: string, target: string) {
    if (source === target) {
      setNotice({ tone: "bad", text: "A node cannot connect to itself." });
      return;
    }
    const targetNode = nodeById.get(target);
    const branch = targetNode?.kind === "auxiliary_classifier" ? "auxiliary" : undefined;
    const duplicate = project.edges.some((edge) => edge.source === source && edge.target === target && edge.branch === branch);
    if (duplicate) {
      setNotice({ tone: "bad", text: "That connection already exists." });
      return;
    }
    commitProject((current) => ({
      ...current,
      edges: [...current.edges, { id: uniqueEdgeId(current, source, target), source, target, branch }],
    }));
    setConnectSourceId(null);
    setNotice({ tone: "good", text: `Connected ${nodeById.get(source)?.name ?? source} to ${targetNode?.name ?? target}.` });
  }

  function removeEdge(edgeId: string) {
    commitProject((current) => ({ ...current, edges: current.edges.filter((edge) => edge.id !== edgeId) }));
    setNotice({ tone: "good", text: "Connection removed." });
  }

  function chooseNode(nodeId: string) {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    if (connectSourceId && connectSourceId !== nodeId) {
      connectNodes(connectSourceId, nodeId);
    }
    setSelectedNodeId(nodeId);
  }

  return (
    <main className="appShell">
      <header className="topbar">
        <div>
          <p className="eyebrow">MNIST topology compiler</p>
          <h1>{project.name}</h1>
        </div>
        <div className="topbarActions">
          <button className="iconButton" type="button" onClick={undo} disabled={history.past.length === 0} title="Undo" aria-label="Undo">
            <Undo2 size={18} />
          </button>
          <button className="iconButton" type="button" onClick={redo} disabled={history.future.length === 0} title="Redo" aria-label="Redo">
            <Redo2 size={18} />
          </button>
          <button className="iconButton" type="button" onClick={exportProject} title="Export project JSON" aria-label="Export project JSON">
            <Download size={18} />
          </button>
          <button
            className="iconButton"
            type="button"
            onClick={() => importInputRef.current?.click()}
            title="Import project JSON"
            aria-label="Import project JSON"
          >
            <Upload size={18} />
          </button>
          <button className="iconButton" type="button" onClick={restoreTemplate} title="Restore template" aria-label="Restore template">
            <RefreshCw size={18} />
          </button>
          <input
            ref={importInputRef}
            className="hiddenInput"
            type="file"
            accept="application/json,.json"
            onChange={(event) => void importProject(event.target.files?.[0])}
          />
        </div>
      </header>
      {notice ? <div className={`notice ${notice.tone}`}>{notice.text}</div> : null}

      <section className="summaryGrid">
        <SummaryItem label="Parameters" value={formatCompactNumber(resolution.totalParameters)} />
        <SummaryItem label="Forward FLOPs" value={formatCompactNumber(resolution.totalFlops)} />
        <SummaryItem label="Branches" value={resolution.branchCount.toLocaleString()} />
        <SummaryItem label="Embedding" value={`${resolution.embeddingDimension ?? "-"}D`} />
        <SummaryItem
          label="Validation"
          value={resolution.errors.length === 0 ? "Pass" : `${resolution.errors.length} issue${resolution.errors.length === 1 ? "" : "s"}`}
          tone={resolution.errors.length === 0 ? "good" : "bad"}
        />
      </section>

      <GeneratedTrainingPanel metrics={trainMetrics} busy={metricsBusy} onRefresh={loadGeneratedMetrics} />

      <section className="workspace">
        <aside className="libraryPanel" aria-label="Node library">
          <div className="panelHeader">
            <strong>Templates</strong>
            <BrainCircuit size={18} />
          </div>
          <button className="templateButton active" type="button" onClick={restoreTemplate}>
            <strong>Enhanced Five-Block MNIST V1</strong>
            <span>5 residual blocks / auxiliary head / GAP + GMP / 128D embedding</span>
          </button>
          <div className="nodePalette" aria-label="Add nodes">
            <button type="button" onClick={() => addNode("multi_branch_residual")}>
              <Plus size={15} />
              Block
            </button>
            <button type="button" onClick={() => addNode("auxiliary_classifier")}>
              <Plus size={15} />
              Aux head
            </button>
            <button type="button" onClick={() => addNode("feature_head")}>
              <Plus size={15} />
              Feature head
            </button>
            <button type="button" onClick={deleteSelectedNode}>
              <Trash2 size={15} />
              Delete
            </button>
          </div>
          {connectSourceId ? (
            <div className="linkNotice">Connecting from {nodeById.get(connectSourceId)?.name ?? connectSourceId}</div>
          ) : null}
        </aside>

        <section className="canvasPanel" aria-label="Topology canvas">
          <div className="canvasViewport">
            <div className="topologyBoard" style={{ width: boardSize.width, height: boardSize.height }}>
              <svg className="topologyEdges" viewBox={`0 0 ${boardSize.width} ${boardSize.height}`} aria-hidden="true">
                {project.edges.map((edge) => {
                  const source = nodeById.get(edge.source);
                  const target = nodeById.get(edge.target);
                  if (!source || !target) {
                    return null;
                  }
                  const sourceX = source.position.x + 158;
                  const sourceY = source.position.y + 44;
                  const targetX = target.position.x;
                  const targetY = target.position.y + 44;
                  const midX = Math.round((sourceX + targetX) / 2);
                  const points =
                    edge.branch === "auxiliary"
                      ? `${sourceX},${sourceY} ${sourceX},${targetY} ${targetX},${targetY}`
                      : `${sourceX},${sourceY} ${midX},${sourceY} ${midX},${targetY} ${targetX},${targetY}`;
                  return <polyline className={edge.branch === "auxiliary" ? "auxiliaryEdge" : ""} key={edge.id} points={points} />;
                })}
              </svg>
              {project.nodes.map((node) => {
                const trace = resolution.trace.find((entry) => entry.nodeId === node.id);
                const selected = selectedNodeId === node.id;
                const invalid = Boolean(trace?.errors.length);
                return (
                  <button
                    className={`topologyNode ${selected ? "selected" : ""} ${invalid ? "invalid" : ""}`}
                    key={node.id}
                    type="button"
                    onClick={() => chooseNode(node.id)}
                    onPointerDown={(event) => startNodeDrag(event, node)}
                    onPointerMove={moveNode}
                    onPointerUp={commitCurrentDrag}
                    onPointerCancel={commitCurrentDrag}
                    style={{ left: node.position.x, top: node.position.y }}
                  >
                    <span>{node.kind.replaceAll("_", " ")}</span>
                    <strong>{node.name}</strong>
                    <small>{formatShape(trace?.outputShape)}</small>
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        <aside className="inspectorPanel" aria-label="Selected node inspector">
          <div className="panelHeader">
            <strong>Inspector</strong>
            {selectedTrace?.errors.length ? <XCircle size={18} /> : <CheckCircle2 size={18} />}
          </div>
          <div className="inspectorTitle">
            <span>{selectedNode.kind.replaceAll("_", " ")}</span>
            <h2>{selectedNode.name}</h2>
            <p>{selectedNode.description}</p>
          </div>
          <div className="shapeGrid">
            <span>Input</span>
            <strong>{formatShape(selectedTrace?.inputShape)}</strong>
            <span>Output</span>
            <strong>{formatShape(selectedTrace?.outputShape)}</strong>
            <span>Params</span>
            <strong>{formatCompactNumber(selectedTrace?.parameters ?? 0)}</strong>
            <span>FLOPs</span>
            <strong>{formatCompactNumber(selectedTrace?.flops ?? 0)}</strong>
          </div>
          <div className="parameterEditor">
            {Object.entries(selectedNode.parameters).map(([key, value]) => (
              <label key={key}>
                <span>{key.replaceAll("_", " ")}</span>
                {typeof value === "boolean" ? (
                  <input
                    type="checkbox"
                    checked={value}
                    onChange={(event) => updateNodeParameter(selectedNode.id, key, event.target.checked)}
                  />
                ) : typeof value === "number" ? (
                  <input
                    type="number"
                    step={key.includes("drop") || key.includes("weight") ? 0.01 : 1}
                    value={value}
                    onChange={(event) => updateNodeParameter(selectedNode.id, key, Number(event.target.value))}
                  />
                ) : key === "mode" ? (
                  <select value={value} onChange={(event) => updateNodeParameter(selectedNode.id, key, event.target.value)}>
                    <option value="gap">GAP</option>
                    <option value="gap_gmp">GAP + GMP</option>
                  </select>
                ) : (
                  <input value={value} onChange={(event) => updateNodeParameter(selectedNode.id, key, event.target.value)} />
                )}
              </label>
            ))}
          </div>
          {branchRepair ? (
            <button className="repairButton" type="button" onClick={repairSelectedBranchWidth}>
              Repair branch split to {branchRepair.repairedWidth} channels
            </button>
          ) : null}
          <div className="edgeEditor">
            <div className="edgeEditorHeader">
              <strong>Connections</strong>
              <button type="button" onClick={() => setConnectSourceId(selectedNode.id)}>
                <Link2 size={15} />
                Start
              </button>
            </div>
            {selectedEdges.length === 0 ? (
              <span className="emptyEdge">No connections yet.</span>
            ) : (
              selectedEdges.map((edge) => (
                <div className="edgeRow" key={edge.id}>
                  <span>
                    {nodeById.get(edge.source)?.name ?? edge.source} {"->"} {nodeById.get(edge.target)?.name ?? edge.target}
                    {edge.branch ? <small>{edge.branch}</small> : null}
                  </span>
                  <button type="button" onClick={() => removeEdge(edge.id)} aria-label={`Remove ${edge.id}`}>
                    <Unlink2 size={15} />
                  </button>
                </div>
              ))
            )}
          </div>
        </aside>
      </section>

      <section className="bottomPanel">
        <div className="panelTabs" aria-label="Compiler panels">
          <PanelButton icon={<Table2 size={17} />} label="Shape Trace" value="trace" selected={bottomPanel} onSelect={setBottomPanel} />
          <PanelButton
            icon={<TriangleAlert size={17} />}
            label="Validation"
            value="validation"
            selected={bottomPanel}
            onSelect={setBottomPanel}
          />
          <PanelButton icon={<FileCode2 size={17} />} label="Generated Files" value="generated" selected={bottomPanel} onSelect={setBottomPanel} />
        </div>

        {bottomPanel === "trace" ? (
          <div className="traceTable">
            <div className="tableHead">Node</div>
            <div className="tableHead">Input</div>
            <div className="tableHead">Output</div>
            <div className="tableHead">Parameters</div>
            {resolution.trace.map((entry) => (
              <Fragment key={entry.nodeId}>
                <div className={entry.errors.length ? "errorText" : undefined}>{entry.name}</div>
                <div>{formatShape(entry.inputShape)}</div>
                <div>{formatShape(entry.outputShape)}</div>
                <div>{formatCompactNumber(entry.parameters)}</div>
              </Fragment>
            ))}
          </div>
        ) : null}

        {bottomPanel === "validation" ? (
          <div className="validationList">
            {resolution.errors.length === 0 ? (
              <div className="validationItem good">
                <CheckCircle2 size={18} />
                <span>Topology passes the current MNIST validation rules.</span>
              </div>
            ) : (
              resolution.errors.map((error) => (
                <div className="validationItem bad" key={error}>
                  <XCircle size={18} />
                  <span>
                    {error}
                    <small>{validationHint(error)}</small>
                  </span>
                </div>
              ))
            )}
          </div>
        ) : null}

        {bottomPanel === "generated" ? (
          <GeneratedFilesPanel
            files={generatedFiles}
            selectedPath={selectedGeneratedFile.path}
            onSelect={setSelectedGeneratedPath}
            onDownload={downloadGeneratedBundle}
          />
        ) : null}
      </section>
    </main>
  );
}

function GeneratedTrainingPanel({
  metrics,
  busy,
  onRefresh,
}: {
  metrics: GeneratedTrainMetrics;
  busy: boolean;
  onRefresh: () => void;
}) {
  return (
    <section className="trainingPanel">
      <div className="trainingPanelHeader">
        <div>
          <p className="eyebrow">Generated runtime</p>
          <h2>MNIST training check</h2>
        </div>
        <button type="button" onClick={onRefresh} disabled={busy}>
          <RefreshCw size={16} />
          Refresh
        </button>
      </div>
      {metrics ? (
        <div className="trainingMetricGrid">
          <span>
            Accuracy <strong>{formatPercent(metrics.test_accuracy)}</strong>
          </span>
          <span>
            Train loss <strong>{metrics.train_loss.toFixed(3)}</strong>
          </span>
          <span>
            Test loss <strong>{metrics.test_loss.toFixed(3)}</strong>
          </span>
          <span>
            Final batch <strong>{metrics.final_batch_loss.toFixed(3)}</strong>
          </span>
          <span>
            Samples <strong>{metrics.train_limit.toLocaleString()} / {metrics.test_limit.toLocaleString()}</strong>
          </span>
          <span>
            Checkpoint <strong>{metrics.checkpoint}</strong>
          </span>
        </div>
      ) : (
        <div className="emptyTraining">No generated training metrics yet.</div>
      )}
    </section>
  );
}

function GeneratedFilesPanel({
  files,
  selectedPath,
  onSelect,
  onDownload,
}: {
  files: GeneratedFile[];
  selectedPath: string;
  onSelect: (path: string) => void;
  onDownload: () => void;
}) {
  const selected = files.find((file) => file.path === selectedPath) ?? files[0];
  return (
    <div className="generatedPanel">
      <div className="generatedHeader">
        <div className="generatedTabs" aria-label="Generated file tabs">
          {files.map((file) => (
            <button className={file.path === selected.path ? "selected" : ""} key={file.path} type="button" onClick={() => onSelect(file.path)}>
              {file.path}
            </button>
          ))}
        </div>
        <button className="downloadBundleButton" type="button" onClick={onDownload}>
          <Download size={16} />
          Bundle
        </button>
      </div>
      <pre className="codePreview">{selected.content}</pre>
    </div>
  );
}

function getBranchRepair(node: { kind: string; parameters: Record<string, boolean | number | string> }) {
  if (node.kind !== "multi_branch_residual") {
    return null;
  }
  const outChannels = typeof node.parameters.out_channels === "number" ? node.parameters.out_channels : undefined;
  const branchCount = typeof node.parameters.branch_count === "number" ? node.parameters.branch_count : undefined;
  if (!outChannels || !branchCount || branchCount < 1 || outChannels % branchCount === 0) {
    return null;
  }
  return { repairedWidth: Math.ceil(outChannels / branchCount) * branchCount };
}

function validationHint(error: string) {
  if (error.includes("equal branches")) {
    return "Use the inspector repair action or choose a width divisible by the branch count.";
  }
  if (error.includes("10 logits")) {
    return "MNIST classifiers need exactly ten class outputs.";
  }
  if (error.includes("spatial dimensions")) {
    return "Reduce stride or insert fewer downsampling operations.";
  }
  return "Edit the selected node parameters, then recheck this panel.";
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function createNode(kind: Extract<NodeKind, "multi_branch_residual" | "auxiliary_classifier" | "feature_head">, id: string, position: TopologyNode["position"]): TopologyNode {
  if (kind === "auxiliary_classifier") {
    return {
      id,
      name: "Auxiliary head",
      kind,
      description: "Auxiliary classifier attached to an intermediate feature tensor.",
      position,
      parameters: { classes: 10 },
    };
  }
  if (kind === "feature_head") {
    return {
      id,
      name: "Feature head",
      kind,
      description: "Projection head for an ensemble embedding.",
      position,
      parameters: { hidden_features: 192, embedding_features: 128, dropout: 0.2 },
    };
  }
  return {
    id,
    name: `Block ${id.replace(/\D/g, "") || ""}`.trim(),
    kind,
    description: "Editable four-branch residual block.",
    position,
    parameters: { out_channels: 128, stride: 1, branch_count: 4, use_se: true, se_reduction: 8, drop_path: 0.02 },
  };
}

function uniqueNodeId(project: TopologyProject, prefix: string) {
  const ids = new Set(project.nodes.map((node) => node.id));
  let index = project.nodes.length + 1;
  let id = `${prefix}${index}`;
  while (ids.has(id)) {
    index += 1;
    id = `${prefix}${index}`;
  }
  return id;
}

function uniqueEdgeId(project: TopologyProject, source: string, target: string) {
  const ids = new Set(project.edges.map((edge) => edge.id));
  const base = `edge_${source}_${target}`;
  let id = base;
  let index = 2;
  while (ids.has(id)) {
    id = `${base}_${index}`;
    index += 1;
  }
  return id;
}

function firstMainOutgoing(project: TopologyProject, nodeId: string): TopologyEdge | undefined {
  const nodeById = new Map(project.nodes.map((node) => [node.id, node]));
  return project.edges.find((edge) => {
    const target = nodeById.get(edge.target);
    return edge.source === nodeId && edge.branch !== "auxiliary" && target?.kind !== "auxiliary_classifier";
  });
}

function getBoardSize(nodes: TopologyNode[]) {
  const maxX = Math.max(...nodes.map((node) => node.position.x), 0);
  const maxY = Math.max(...nodes.map((node) => node.position.y), 0);
  return {
    width: Math.max(2040, maxX + 220),
    height: Math.max(390, maxY + 140),
  };
}

function SummaryItem({ label, value, tone }: { label: string; value: string; tone?: "good" | "bad" }) {
  return (
    <div className={`summaryItem ${tone ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PanelButton({
  icon,
  label,
  value,
  selected,
  onSelect,
}: {
  icon: ReactNode;
  label: string;
  value: BottomPanel;
  selected: BottomPanel;
  onSelect: (value: BottomPanel) => void;
}) {
  return (
    <button className={selected === value ? "selected" : ""} type="button" onClick={() => onSelect(value)}>
      {icon}
      {label}
    </button>
  );
}
