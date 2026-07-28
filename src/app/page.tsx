"use client";

import {
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  Download,
  FileCode2,
  FolderOpen,
  Link2,
  ListChecks,
  LoaderCircle,
  Plus,
  Play,
  Redo2,
  RefreshCw,
  Save,
  ScrollText,
  Table2,
  TriangleAlert,
  Trash2,
  Trophy,
  Undo2,
  Unlink2,
  Upload,
  XCircle,
} from "lucide-react";
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type PointerEvent, type ReactNode } from "react";

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
import { topologyVersionId } from "@/lib/topologyVersion";

type BottomPanel = "trace" | "validation" | "generated";
type EditorHistory = { past: TopologyProject[]; present: TopologyProject; future: TopologyProject[] };
type Notice = { tone: "good" | "bad"; text: string } | null;
type TrainSettings = {
  epochs: number;
  trainLimit: number;
  testLimit: number;
  batchSize: number;
  cpu: boolean;
};
type QueuePreset = "width" | "dropPath" | "pooling" | "se";
type GeneratedTrainMetrics = {
  status: string;
  run_id?: string;
  topology_id?: string;
  created_at?: string;
  updated_at?: string;
  epochs: number;
  train_limit: number;
  test_limit: number;
  seed?: number;
  current_epoch?: number;
  current_batch?: number;
  total_batches?: number;
  first_batch_loss: number | null;
  final_batch_loss: number | null;
  final_batch_accuracy?: number | null;
  train_loss: number | null;
  train_accuracy?: number | null;
  baseline_test_loss?: number | null;
  baseline_accuracy?: number | null;
  test_loss: number | null;
  test_accuracy: number | null;
  best_accuracy?: number | null;
  best_epoch?: number | null;
  accuracy_delta?: number | null;
  epoch_history?: Array<{ epoch: number; train_loss: number; test_loss: number; test_accuracy: number }>;
  checkpoint?: string;
  duration_seconds?: number;
  passed_smoke_rule: boolean | null;
  error?: string;
} | null;
type GeneratedRunSummary = {
  runId: string;
  topologyId?: string;
  createdAt?: string;
  updatedAt?: string;
  status: string;
  metrics: NonNullable<GeneratedTrainMetrics>;
};
type ExperimentSummary = {
  name: string;
  preset: string;
  updatedAt: string;
  bestVariantId: string;
  variantCount: number;
};
type ExperimentSession = {
  name: string;
  displayName?: string;
  preset: QueuePreset;
  createdAt: string;
  updatedAt: string;
  bestVariantId: string;
  queue: AblationQueueItem[];
  reportRows: ReturnType<typeof buildQueueReportRows>;
};
type AblationQueueItem = {
  id: string;
  label: string;
  note: string;
  project: TopologyProject;
  topologyId: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  runId?: string;
  metrics?: NonNullable<GeneratedTrainMetrics>;
  error?: string;
};
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
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [runHistory, setRunHistory] = useState<GeneratedRunSummary[]>([]);
  const [runsBusy, setRunsBusy] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [ablationQueue, setAblationQueue] = useState<AblationQueueItem[]>([]);
  const [queueBusy, setQueueBusy] = useState(false);
  const [queuePreset, setQueuePreset] = useState<QueuePreset>("width");
  const [experimentName, setExperimentName] = useState("width-sweep-001");
  const [savedExperiments, setSavedExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExperimentName, setSelectedExperimentName] = useState("");
  const [experimentsBusy, setExperimentsBusy] = useState(false);
  const [trainSettings, setTrainSettings] = useState<TrainSettings>({
    epochs: 1,
    trainLimit: 1024,
    testLimit: 512,
    batchSize: 128,
    cpu: true,
  });
  const [connectSourceId, setConnectSourceId] = useState<string | null>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const dragRef = useRef<DragState | null>(null);
  const queueCancelRef = useRef(false);
  const suppressClickRef = useRef(false);
  const project = history.present;
  const resolution = useMemo(() => resolveTopology(project), [project]);
  const nodeById = useMemo(() => new Map(project.nodes.map((node) => [node.id, node])), [project.nodes]);
  const selectedNode = project.nodes.find((node) => node.id === selectedNodeId) ?? project.nodes[0];
  const selectedTrace = resolution.trace.find((entry) => entry.nodeId === selectedNode.id);
  const selectedEdges = project.edges.filter((edge) => edge.source === selectedNode.id || edge.target === selectedNode.id);
  const generatedFiles = useMemo(() => generateFiles(project), [project]);
  const selectedGeneratedFile = generatedFiles.find((file) => file.path === selectedGeneratedPath) ?? generatedFiles[0];
  const currentTopologyId = useMemo(() => topologyVersionId(project), [project]);
  const branchRepair = getBranchRepair(selectedNode);
  const boardSize = useMemo(() => getBoardSize(project.nodes), [project.nodes]);

  const loadGeneratedMetrics = useCallback(async () => {
    setMetricsBusy(true);
    try {
      const response = await fetch("/api/generated-metrics", { cache: "no-store" });
      const payload = await response.json();
      setTrainMetrics(payload.metrics ?? null);
    } finally {
      setMetricsBusy(false);
    }
  }, []);

  const loadGeneratedRuns = useCallback(async () => {
    setRunsBusy(true);
    try {
      const response = await fetch("/api/generated-runs", { cache: "no-store" });
      const payload = await response.json();
      setRunHistory(Array.isArray(payload.runs) ? payload.runs : []);
    } finally {
      setRunsBusy(false);
    }
  }, []);

  const loadExperiments = useCallback(async () => {
    setExperimentsBusy(true);
    try {
      const response = await fetch("/api/experiments", { cache: "no-store" });
      const payload = await response.json();
      setSavedExperiments(Array.isArray(payload.experiments) ? payload.experiments : []);
    } finally {
      setExperimentsBusy(false);
    }
  }, []);

  useEffect(() => {
    void loadGeneratedMetrics();
    void loadGeneratedRuns();
    void loadExperiments();
  }, [loadExperiments, loadGeneratedMetrics, loadGeneratedRuns]);

  useEffect(() => {
    if (!trainingBusy || !activeRunId || queueBusy) {
      return;
    }

    const runId = activeRunId;
    let cancelled = false;
    async function pollRun() {
      try {
        const response = await fetch(`/api/train-generated?runId=${encodeURIComponent(runId)}`, { cache: "no-store" });
        const payload = await response.json();
        if (cancelled) {
          return;
        }
        if (payload.metrics) {
          setTrainMetrics(payload.metrics);
        }
        if (Array.isArray(payload.logs)) {
          setTrainingLogs(payload.logs);
        }
        if (payload.status === "complete" || payload.status === "failed") {
          setTrainingBusy(false);
          setNotice({
            tone: payload.status === "complete" ? "good" : "bad",
            text: payload.status === "complete" ? "Generated training completed." : (payload.error ?? "Generated training failed."),
          });
          void loadGeneratedMetrics();
          void loadGeneratedRuns();
        }
      } catch (error) {
        if (!cancelled) {
          setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Could not read the live training run." });
        }
      }
    }

    void pollRun();
    const interval = window.setInterval(() => void pollRun(), 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeRunId, loadGeneratedMetrics, loadGeneratedRuns, queueBusy, trainingBusy]);

  async function startGeneratedRun(targetProject: TopologyProject) {
    const response = await fetch("/api/train-generated", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project: targetProject, ...trainSettings }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.error ?? `Training failed with status ${response.status}.`);
    }
    setActiveRunId(payload.runId ?? null);
    setTrainMetrics(payload.metrics ?? null);
    setTrainingLogs(Array.isArray(payload.logs) ? payload.logs : []);
    setSelectedRunId(payload.runId ?? null);
    return payload as { runId: string; metrics?: NonNullable<GeneratedTrainMetrics>; logs?: string[]; status?: string; error?: string };
  }

  async function waitForGeneratedRun(runId: string) {
    for (;;) {
      await sleep(1000);
      const response = await fetch(`/api/train-generated?runId=${encodeURIComponent(runId)}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error ?? `Could not read run ${runId}.`);
      }
      if (payload.metrics) {
        setTrainMetrics(payload.metrics);
      }
      if (Array.isArray(payload.logs)) {
        setTrainingLogs(payload.logs);
      }
      if (payload.status === "complete" || payload.status === "failed" || payload.status === "cancelled") {
        return payload as {
          status: "complete" | "failed" | "cancelled";
          metrics?: NonNullable<GeneratedTrainMetrics>;
          logs?: string[];
          error?: string;
        };
      }
    }
  }

  async function cancelGeneratedRun(runId: string) {
    const response = await fetch(`/api/train-generated?runId=${encodeURIComponent(runId)}`, { method: "DELETE" });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.error ?? `Could not cancel run ${runId}.`);
    }
    if (payload?.metrics) {
      setTrainMetrics(payload.metrics);
    }
    if (Array.isArray(payload?.logs)) {
      setTrainingLogs(payload.logs);
    }
    return payload as { status: "cancelled"; metrics?: NonNullable<GeneratedTrainMetrics>; logs?: string[]; error?: string };
  }

  async function runGeneratedTraining() {
    setTrainingBusy(true);
    setTrainingLogs([]);
    setActiveRunId(null);
    setNotice({ tone: "good", text: "Generated training started." });
    try {
      await startGeneratedRun(project);
      setNotice({ tone: "good", text: "Generated training is running." });
    } catch (error) {
      setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Generated training failed." });
      setTrainingBusy(false);
    } finally {
      void loadGeneratedRuns();
    }
  }

  function buildAblationQueue() {
    const variants = createAblationVariants(project, queuePreset);
    setAblationQueue(variants);
    if (!experimentName.trim()) {
      setExperimentName(defaultExperimentName(queuePreset));
    }
    setNotice({ tone: "good", text: `Queued ${variants.length} topology variants.` });
  }

  async function runAblationQueue() {
    const preparedQueue = ablationQueue.length > 0 ? ablationQueue : createAblationVariants(project, queuePreset);
    let latestQueue = preparedQueue;
    setAblationQueue(preparedQueue);
    queueCancelRef.current = false;
    setQueueBusy(true);
    setTrainingBusy(true);
    setTrainingLogs([]);
    setNotice({ tone: "good", text: "Ablation queue started." });
    try {
      for (const item of preparedQueue) {
        if (queueCancelRef.current) {
          break;
        }
        if (item.status === "complete") {
          continue;
        }
        latestQueue = updateQueueItem(latestQueue, item.id, { status: "running", error: undefined });
        setAblationQueue(latestQueue);
        const started = await startGeneratedRun(item.project);
        latestQueue = updateQueueItem(latestQueue, item.id, { runId: started.runId });
        setAblationQueue(latestQueue);
        const finished = await waitForGeneratedRun(started.runId);
        latestQueue = updateQueueItem(latestQueue, item.id, {
          status: finished.status,
          metrics: finished.metrics,
          error: finished.error,
        });
        setAblationQueue(latestQueue);
        void loadGeneratedRuns();
        if (finished.status === "cancelled") {
          break;
        }
      }
      if (latestQueue.length > 0) {
        await saveExperimentSnapshot(latestQueue);
      }
      setNotice({ tone: queueCancelRef.current ? "bad" : "good", text: queueCancelRef.current ? "Ablation queue stopped." : "Ablation queue completed." });
    } catch (error) {
      setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Ablation queue failed." });
    } finally {
      setQueueBusy(false);
      setTrainingBusy(false);
      void loadGeneratedMetrics();
      void loadGeneratedRuns();
    }
  }

  async function stopQueueOrRun() {
    queueCancelRef.current = true;
    try {
      if (activeRunId) {
        await cancelGeneratedRun(activeRunId);
      }
      setAblationQueue((current) =>
        current.map((item) => (item.status === "complete" ? item : { ...item, status: item.status === "running" ? "cancelled" : item.status })),
      );
      setNotice({ tone: "bad", text: activeRunId ? "Current generated run cancelled." : "Queue stop requested." });
    } catch (error) {
      setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Could not cancel the current run." });
    } finally {
      setQueueBusy(false);
      setTrainingBusy(false);
      void loadGeneratedMetrics();
      void loadGeneratedRuns();
    }
  }

  function clearQueuedVariants() {
    setAblationQueue((current) => current.filter((item) => item.status === "complete"));
    setNotice({ tone: "good", text: "Queued variants cleared. Completed results kept." });
  }

  function loadQueueVariant(item: AblationQueueItem) {
    commitProject(item.project);
    const nextSelected = item.project.nodes.some((node) => node.id === selectedNodeId) ? selectedNodeId : item.project.nodes[0]?.id;
    if (nextSelected) {
      setSelectedNodeId(nextSelected);
    }
    setNotice({ tone: "good", text: `Loaded ${item.label} onto the canvas.` });
  }

  function exportQueueReport(format: "json" | "csv") {
    const rows = buildQueueReportRows(ablationQueue);
    if (rows.length === 0) {
      setNotice({ tone: "bad", text: "No queue variants to export yet." });
      return;
    }
    if (format === "json") {
      downloadTextFile("mnist-ablation-report.json", JSON.stringify({ exportedAt: new Date().toISOString(), rows }, null, 2), "application/json");
      setNotice({ tone: "good", text: "JSON comparison report exported." });
      return;
    }
    downloadTextFile("mnist-ablation-report.csv", toCsv(rows), "text/csv");
    setNotice({ tone: "good", text: "CSV comparison report exported." });
  }

  async function saveExperimentSnapshot(queue = ablationQueue) {
    if (queue.length === 0) {
      setNotice({ tone: "bad", text: "Build a queue before saving an experiment." });
      return;
    }
    const best = bestQueueVariant(queue);
    const name = experimentName.trim() || defaultExperimentName(queuePreset);
    setExperimentName(name);
    const response = await fetch("/api/experiments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        preset: queuePreset,
        queue,
        bestVariantId: best?.id ?? "",
        reportRows: buildQueueReportRows(queue),
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.error ?? "Could not save the experiment.");
    }
    setSelectedExperimentName(payload.experiment?.name ?? name);
    await loadExperiments();
    setNotice({ tone: "good", text: `Saved experiment ${payload.experiment?.name ?? name}.` });
  }

  async function loadSavedExperiment(name: string) {
    if (!name) {
      return;
    }
    setExperimentsBusy(true);
    try {
      const response = await fetch(`/api/experiments?name=${encodeURIComponent(name)}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.error ?? "Could not load the experiment.");
      }
      const experiment = payload.experiment as ExperimentSession;
      const queue = hydrateExperimentQueue(experiment.queue);
      setExperimentName(experiment.displayName ?? experiment.name);
      setSelectedExperimentName(experiment.name);
      setQueuePreset(isQueuePreset(experiment.preset) ? experiment.preset : "width");
      setAblationQueue(queue);
      setNotice({ tone: "good", text: `Loaded experiment ${experiment.displayName ?? experiment.name}.` });
    } catch (error) {
      setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Could not load the experiment." });
    } finally {
      setExperimentsBusy(false);
    }
  }

  function loadBestVariant() {
    const best = bestQueueVariant(ablationQueue);
    if (!best) {
      setNotice({ tone: "bad", text: "No completed queue result has an accuracy yet." });
      return;
    }
    loadQueueVariant(best);
    setNotice({ tone: "good", text: `Loaded best variant: ${best.label}.` });
  }

  function changeQueuePreset(preset: QueuePreset) {
    setQueuePreset(preset);
    if (!experimentName.trim()) {
      setExperimentName(defaultExperimentName(preset));
    }
  }

  function refreshTrainingState() {
    void loadGeneratedMetrics();
    void loadGeneratedRuns();
  }

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

      <GeneratedTrainingPanel
        metrics={trainMetrics}
        metricsBusy={metricsBusy}
        trainingBusy={trainingBusy}
        activeRunId={activeRunId}
        logs={trainingLogs}
        runHistory={runHistory}
        runsBusy={runsBusy}
        selectedRunId={selectedRunId}
        currentTopologyId={currentTopologyId}
        ablationQueue={ablationQueue}
        queueBusy={queueBusy}
        queuePreset={queuePreset}
        experimentName={experimentName}
        savedExperiments={savedExperiments}
        selectedExperimentName={selectedExperimentName}
        experimentsBusy={experimentsBusy}
        settings={trainSettings}
        onSettingsChange={setTrainSettings}
        onExperimentNameChange={setExperimentName}
        onQueuePresetChange={changeQueuePreset}
        onLoadExperiment={loadSavedExperiment}
        onSaveExperiment={() => void saveExperimentSnapshot()}
        onLoadBest={loadBestVariant}
        onRefresh={refreshTrainingState}
        onRun={runGeneratedTraining}
        onBuildQueue={buildAblationQueue}
        onRunQueue={runAblationQueue}
        onStopQueue={stopQueueOrRun}
        onClearQueue={clearQueuedVariants}
        onExportReport={exportQueueReport}
        onLoadVariant={loadQueueVariant}
        onSelectRun={setSelectedRunId}
      />

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
  metricsBusy,
  trainingBusy,
  activeRunId,
  logs,
  runHistory,
  runsBusy,
  selectedRunId,
  currentTopologyId,
  ablationQueue,
  queueBusy,
  queuePreset,
  experimentName,
  savedExperiments,
  selectedExperimentName,
  experimentsBusy,
  settings,
  onSettingsChange,
  onExperimentNameChange,
  onQueuePresetChange,
  onLoadExperiment,
  onSaveExperiment,
  onLoadBest,
  onRefresh,
  onRun,
  onBuildQueue,
  onRunQueue,
  onStopQueue,
  onClearQueue,
  onExportReport,
  onLoadVariant,
  onSelectRun,
}: {
  metrics: GeneratedTrainMetrics;
  metricsBusy: boolean;
  trainingBusy: boolean;
  activeRunId: string | null;
  logs: string[];
  runHistory: GeneratedRunSummary[];
  runsBusy: boolean;
  selectedRunId: string | null;
  currentTopologyId: string;
  ablationQueue: AblationQueueItem[];
  queueBusy: boolean;
  queuePreset: QueuePreset;
  experimentName: string;
  savedExperiments: ExperimentSummary[];
  selectedExperimentName: string;
  experimentsBusy: boolean;
  settings: TrainSettings;
  onSettingsChange: (settings: TrainSettings) => void;
  onExperimentNameChange: (name: string) => void;
  onQueuePresetChange: (preset: QueuePreset) => void;
  onLoadExperiment: (name: string) => void;
  onSaveExperiment: () => void;
  onLoadBest: () => void;
  onRefresh: () => void;
  onRun: () => void;
  onBuildQueue: () => void;
  onRunQueue: () => void;
  onStopQueue: () => void;
  onClearQueue: () => void;
  onExportReport: (format: "json" | "csv") => void;
  onLoadVariant: (item: AblationQueueItem) => void;
  onSelectRun: (runId: string | null) => void;
}) {
  function updateSetting<K extends keyof TrainSettings>(key: K, value: TrainSettings[K]) {
    onSettingsChange({ ...settings, [key]: value });
  }

  const selectedRun = runHistory.find((run) => run.runId === selectedRunId);
  const displayedMetrics = selectedRun?.metrics ?? metrics;
  const progress =
    displayedMetrics?.total_batches && displayedMetrics.total_batches > 0
      ? Math.min(100, Math.round(((displayedMetrics.current_batch ?? 0) / displayedMetrics.total_batches) * 100))
      : displayedMetrics?.status === "complete"
        ? 100
        : 0;
  const topologyMatches = displayedMetrics?.topology_id ? displayedMetrics.topology_id === currentTopologyId : undefined;
  const bestAccuracy = displayedMetrics?.best_accuracy ?? displayedMetrics?.test_accuracy;
  const bestEpoch = displayedMetrics?.best_epoch ?? undefined;
  const bestAccuracySuffix = bestEpoch === 0 ? " / base" : typeof bestEpoch === "number" ? ` / e${bestEpoch}` : "";
  const progressEpoch = displayedMetrics?.current_epoch ?? 0;
  const progressBatch = displayedMetrics?.current_batch ?? 0;
  const progressTotal = displayedMetrics?.total_batches ?? 0;
  const chartRuns = buildChartRuns(runHistory, displayedMetrics, ablationQueue);
  const batchLosses = parseBatchLosses(logs);

  return (
    <section className="trainingPanel">
      <div className="trainingPanelHeader">
        <div>
          <p className="eyebrow">Generated runtime</p>
          <h2>MNIST training check</h2>
        </div>
        <div className="trainingActions">
          <button type="button" onClick={onRun} disabled={trainingBusy || queueBusy}>
            {trainingBusy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
            Run
          </button>
          <button type="button" onClick={onBuildQueue} disabled={trainingBusy || queueBusy}>
            <ListChecks size={16} />
            Queue
          </button>
          <button type="button" onClick={onRunQueue} disabled={trainingBusy || queueBusy}>
            {queueBusy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
            Run queue
          </button>
          <button type="button" onClick={onStopQueue} disabled={!trainingBusy && !queueBusy}>
            <XCircle size={16} />
            Stop
          </button>
          <button type="button" onClick={onRefresh} disabled={metricsBusy || trainingBusy}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>
      </div>
      <div className="runStatusStrip">
        <span>
          Status <strong>{displayedMetrics?.status ?? "idle"}</strong>
        </span>
        <span>
          Current topology <strong>{currentTopologyId}</strong>
        </span>
        <span>
          Run <strong>{activeRunId ?? displayedMetrics?.run_id ?? "-"}</strong>
        </span>
        <span className={topologyMatches === false ? "mismatch" : "match"}>
          Match <strong>{topologyMatches === undefined ? "-" : topologyMatches ? "yes" : "no"}</strong>
        </span>
      </div>
      {displayedMetrics ? (
        <div className="trainingProgressBlock" aria-label="Training progress">
          <div className="trainingProgressMeta">
            <span>
              Progress <strong>{progress}%</strong>
            </span>
            <span>
              Epoch <strong>{progressEpoch}/{displayedMetrics.epochs}</strong>
            </span>
            <span>
              Batch <strong>{progressBatch}/{progressTotal}</strong>
            </span>
          </div>
          <div className="trainingProgress">
            <span style={{ width: `${progress}%` }} />
          </div>
        </div>
      ) : null}
      <div className="trainingControls">
        <label>
          <span>Epochs</span>
          <input min={1} max={5} type="number" value={settings.epochs} onChange={(event) => updateSetting("epochs", Number(event.target.value))} />
        </label>
        <label>
          <span>Train samples</span>
          <input
            min={128}
            max={10000}
            step={128}
            type="number"
            value={settings.trainLimit}
            onChange={(event) => updateSetting("trainLimit", Number(event.target.value))}
          />
        </label>
        <label>
          <span>Test samples</span>
          <input
            min={128}
            max={5000}
            step={128}
            type="number"
            value={settings.testLimit}
            onChange={(event) => updateSetting("testLimit", Number(event.target.value))}
          />
        </label>
        <label>
          <span>Batch</span>
          <input
            min={16}
            max={512}
            step={16}
            type="number"
            value={settings.batchSize}
            onChange={(event) => updateSetting("batchSize", Number(event.target.value))}
          />
        </label>
        <label className="trainingToggle">
          <span>CPU</span>
          <input type="checkbox" checked={settings.cpu} onChange={(event) => updateSetting("cpu", event.target.checked)} />
        </label>
      </div>
      <p className="trainingNote">Default settings are tuned for a quick real-MNIST compiler check.</p>
      <QueueToolbar
        preset={queuePreset}
        experimentName={experimentName}
        selectedExperimentName={selectedExperimentName}
        savedExperiments={savedExperiments}
        queueLength={ablationQueue.length}
        queueBusy={queueBusy}
        experimentsBusy={experimentsBusy}
        onExperimentNameChange={onExperimentNameChange}
        onPresetChange={onQueuePresetChange}
        onLoadExperiment={onLoadExperiment}
        onSaveExperiment={onSaveExperiment}
        onLoadBest={onLoadBest}
        onBuildQueue={onBuildQueue}
        onClearQueue={onClearQueue}
        onExportReport={onExportReport}
      />
      {displayedMetrics ? (
        <div className="trainingMetricGrid">
          <span>
            Final acc <strong>{formatMaybePercent(displayedMetrics.test_accuracy)}</strong>
          </span>
          <span>
            Initial acc <strong>{formatMaybePercent(displayedMetrics.baseline_accuracy)}</strong>
          </span>
          <span>
            Best acc <strong>{formatMaybePercent(bestAccuracy)}{bestAccuracySuffix}</strong>
          </span>
          <span>
            Delta <strong>{formatMaybeSignedPercent(displayedMetrics.accuracy_delta)}</strong>
          </span>
          <span>
            Train acc <strong>{formatMaybePercent(displayedMetrics.train_accuracy)}</strong>
          </span>
          <span>
            Train loss <strong>{formatMaybeNumber(displayedMetrics.train_loss)}</strong>
          </span>
          <span>
            Test loss <strong>{formatMaybeNumber(displayedMetrics.test_loss)}</strong>
          </span>
          <span>
            Final batch <strong>{formatMaybeNumber(displayedMetrics.final_batch_loss)}</strong>
          </span>
          <span>
            Samples <strong>{displayedMetrics.train_limit.toLocaleString()} / {displayedMetrics.test_limit.toLocaleString()}</strong>
          </span>
          <span>
            Checkpoint <strong>{displayedMetrics.checkpoint ?? "-"}</strong>
          </span>
        </div>
      ) : (
        <div className="emptyTraining">No generated training metrics yet.</div>
      )}
      <MetricCharts runs={chartRuns} batchLosses={batchLosses} />
      <AblationQueuePanel queue={ablationQueue} currentTopologyId={currentTopologyId} onLoadVariant={onLoadVariant} />
      <div className="trainingRuntimeGrid">
        <div className="trainingLogPanel">
          <div className="subPanelHeader">
            <strong>Live log</strong>
            <ScrollText size={16} />
          </div>
          <pre className="trainingLog">{logs.length > 0 ? logs.join("\n") : "No live log yet."}</pre>
        </div>
        <div className="runHistoryPanel">
          <div className="subPanelHeader">
            <strong>Run history</strong>
            <span>{runsBusy ? "Refreshing" : `${runHistory.length} saved`}</span>
          </div>
          <div className="runHistoryTable">
            <div className="tableHead">Run</div>
            <div className="tableHead">Topology</div>
            <div className="tableHead">Accuracy</div>
            <div className="tableHead">Status</div>
            {runHistory.length === 0 ? (
              <button className="runHistoryEmpty" type="button" onClick={() => onSelectRun(null)}>
                No saved runs yet.
              </button>
            ) : (
              runHistory.slice(0, 8).map((run) => (
                <Fragment key={run.runId}>
                  <button className={`runHistoryCell ${run.runId === selectedRunId ? "selected" : ""}`} type="button" onClick={() => onSelectRun(run.runId)}>
                    {formatRunLabel(run.runId)}
                  </button>
                  <button
                    className={`runHistoryCell topology ${run.topologyId === currentTopologyId ? "match" : "mismatch"} ${run.runId === selectedRunId ? "selected" : ""}`}
                    type="button"
                    onClick={() => onSelectRun(run.runId)}
                  >
                    {run.topologyId ?? "-"}
                  </button>
                  <button className={`runHistoryCell ${run.runId === selectedRunId ? "selected" : ""}`} type="button" onClick={() => onSelectRun(run.runId)}>
                    {formatMaybePercent(run.metrics.best_accuracy ?? run.metrics.test_accuracy)}
                  </button>
                  <button className={`runHistoryCell ${run.runId === selectedRunId ? "selected" : ""}`} type="button" onClick={() => onSelectRun(run.runId)}>
                    {run.status}
                  </button>
                </Fragment>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

function QueueToolbar({
  preset,
  experimentName,
  selectedExperimentName,
  savedExperiments,
  queueLength,
  queueBusy,
  experimentsBusy,
  onExperimentNameChange,
  onPresetChange,
  onLoadExperiment,
  onSaveExperiment,
  onLoadBest,
  onBuildQueue,
  onClearQueue,
  onExportReport,
}: {
  preset: QueuePreset;
  experimentName: string;
  selectedExperimentName: string;
  savedExperiments: ExperimentSummary[];
  queueLength: number;
  queueBusy: boolean;
  experimentsBusy: boolean;
  onExperimentNameChange: (name: string) => void;
  onPresetChange: (preset: QueuePreset) => void;
  onLoadExperiment: (name: string) => void;
  onSaveExperiment: () => void;
  onLoadBest: () => void;
  onBuildQueue: () => void;
  onClearQueue: () => void;
  onExportReport: (format: "json" | "csv") => void;
}) {
  return (
    <div className="queueToolbar">
      <label>
        <span>Experiment</span>
        <input value={experimentName} onChange={(event) => onExperimentNameChange(event.target.value)} disabled={queueBusy} />
      </label>
      <label>
        <span>Preset</span>
        <select value={preset} onChange={(event) => onPresetChange(event.target.value as QueuePreset)} disabled={queueBusy}>
          <option value="width">Width sweep</option>
          <option value="dropPath">Drop-path sweep</option>
          <option value="pooling">Pooling mode sweep</option>
          <option value="se">SE on/off sweep</option>
        </select>
      </label>
      <label>
        <span>Saved</span>
        <select value={selectedExperimentName} onChange={(event) => onLoadExperiment(event.target.value)} disabled={queueBusy || experimentsBusy}>
          <option value="">Load session</option>
          {savedExperiments.map((experiment) => (
            <option key={experiment.name} value={experiment.name}>
              {experiment.name} ({experiment.variantCount})
            </option>
          ))}
        </select>
      </label>
      <button type="button" onClick={onBuildQueue} disabled={queueBusy}>
        <ListChecks size={15} />
        Build
      </button>
      <button type="button" onClick={onSaveExperiment} disabled={queueBusy || queueLength === 0}>
        <Save size={15} />
        Save
      </button>
      <button type="button" onClick={onLoadBest} disabled={queueLength === 0}>
        <Trophy size={15} />
        Best
      </button>
      <button type="button" onClick={() => selectedExperimentName && onLoadExperiment(selectedExperimentName)} disabled={queueBusy || !selectedExperimentName}>
        <FolderOpen size={15} />
        Reload
      </button>
      <button type="button" onClick={onClearQueue} disabled={queueBusy || queueLength === 0}>
        <Trash2 size={15} />
        Clear queued
      </button>
      <button type="button" onClick={() => onExportReport("json")} disabled={queueLength === 0}>
        <Download size={15} />
        JSON
      </button>
      <button type="button" onClick={() => onExportReport("csv")} disabled={queueLength === 0}>
        <Download size={15} />
        CSV
      </button>
    </div>
  );
}

function MetricCharts({ runs, batchLosses }: { runs: Array<{ label: string; accuracy?: number; trainLoss?: number; testLoss?: number }>; batchLosses: number[] }) {
  return (
    <div className="metricChartsPanel">
      <div className="subPanelHeader">
        <strong>Metric charts</strong>
        <BarChart3 size={16} />
      </div>
      <div className="metricChartGrid">
        <TinyLineChart title="Accuracy" tone="good" values={runs.map((run) => run.accuracy)} labels={runs.map((run) => run.label)} formatValue={formatMaybePercent} />
        <TinyLineChart title="Train loss" tone="blue" values={runs.map((run) => run.trainLoss)} labels={runs.map((run) => run.label)} formatValue={formatMaybeNumber} invert />
        <TinyLineChart title="Test loss" tone="amber" values={runs.map((run) => run.testLoss)} labels={runs.map((run) => run.label)} formatValue={formatMaybeNumber} invert />
        <TinyLineChart title="Live batch loss" tone="red" values={batchLosses} labels={batchLosses.map((_, index) => `B${index + 1}`)} formatValue={formatMaybeNumber} invert />
      </div>
    </div>
  );
}

function TinyLineChart({
  title,
  values,
  labels,
  formatValue,
  tone,
  invert = false,
}: {
  title: string;
  values: Array<number | null | undefined>;
  labels: string[];
  formatValue: (value: number | null | undefined) => string;
  tone: "good" | "blue" | "amber" | "red";
  invert?: boolean;
}) {
  const points = values
    .map((value, index) => (typeof value === "number" && Number.isFinite(value) ? { value, index } : null))
    .filter((point): point is { value: number; index: number } => Boolean(point));
  const width = 260;
  const height = 86;
  const pad = 12;
  const min = Math.min(...points.map((point) => point.value), 0);
  const max = Math.max(...points.map((point) => point.value), 1);
  const range = max - min || 1;
  const path = points
    .map((point, order) => {
      const x = pad + (points.length === 1 ? width / 2 - pad : (point.index / Math.max(1, values.length - 1)) * (width - pad * 2));
      const normalized = (point.value - min) / range;
      const y = pad + (invert ? normalized : 1 - normalized) * (height - pad * 2);
      return `${order === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" " );
  const latest = points.at(-1);
  return (
    <div className={`metricChart ${tone}`}>
      <div>
        <span>{title}</span>
        <strong>{formatValue(latest?.value)}</strong>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
        {path ? <path d={path} /> : null}
        {points.map((point) => {
          const x = pad + (points.length === 1 ? width / 2 - pad : (point.index / Math.max(1, values.length - 1)) * (width - pad * 2));
          const normalized = (point.value - min) / range;
          const y = pad + (invert ? normalized : 1 - normalized) * (height - pad * 2);
          return <circle key={`${point.index}-${point.value}`} cx={x} cy={y} r="3"><title>{`${labels[point.index] ?? point.index}: ${formatValue(point.value)}`}</title></circle>;
        })}
      </svg>
    </div>
  );
}

function AblationQueuePanel({
  queue,
  currentTopologyId,
  onLoadVariant,
}: {
  queue: AblationQueueItem[];
  currentTopologyId: string;
  onLoadVariant: (item: AblationQueueItem) => void;
}) {
  if (queue.length === 0) {
    return null;
  }
  return (
    <div className="ablationQueuePanel">
      <div className="subPanelHeader">
        <strong>Ablation queue</strong>
        <span>{queue.length} variants</span>
      </div>
      <div className="ablationQueueGrid">
        {queue.map((item) => (
          <button className={`ablationQueueItem ${item.status}`} key={item.id} type="button" onClick={() => onLoadVariant(item)}>
            <span>{item.label}</span>
            <strong>{formatMaybePercent(item.metrics?.best_accuracy ?? item.metrics?.test_accuracy)}</strong>
            <small>{item.note}</small>
            <small>train {formatMaybeNumber(item.metrics?.train_loss)} / test {formatMaybeNumber(item.metrics?.test_loss)}</small>
            <small>{item.topologyId === currentTopologyId ? "current" : item.topologyId} / {item.status}</small>
          </button>
        ))}
      </div>
    </div>
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

function createAblationVariants(project: TopologyProject, preset: QueuePreset): AblationQueueItem[] {
  const variants = [baseVariant(project), ...presetVariants(project, preset)];
  return variants.map((variant) => ({
    ...variant,
    topologyId: topologyVersionId(variant.project),
    status: "queued" as const,
  }));
}

function baseVariant(project: TopologyProject) {
  return {
    id: "current",
    label: "Current",
    note: "Canvas topology",
    project,
  };
}

function presetVariants(project: TopologyProject, preset: QueuePreset) {
  if (preset === "width") {
    const block5 = project.nodes.find((node) => node.id === "block5");
    const baseWidth = typeof block5?.parameters.out_channels === "number" ? block5.parameters.out_channels : 160;
    return [-32, -16, 16, 32].map((delta) => {
      const width = Math.max(16, nearestMultiple(baseWidth + delta, 4));
      return {
        id: `width-${width}`,
        label: `${width} ch`,
        note: `Block 5 width ${delta > 0 ? "+" : ""}${delta}`,
        project: mutateProject(project, `Width ${width}`, (node) =>
          node.id === "block5" ? { ...node, parameters: { ...node.parameters, out_channels: width } } : node,
        ),
      };
    });
  }

  if (preset === "dropPath") {
    return [
      { id: "drop-0", label: "No drop", factor: 0 },
      { id: "drop-half", label: "Half drop", factor: 0.5 },
      { id: "drop-base", label: "Base drop", factor: 1 },
      { id: "drop-high", label: "High drop", factor: 1.5 },
    ].map((item) => ({
      id: item.id,
      label: item.label,
      note: item.factor === 0 ? "Drop path disabled" : `Drop path x${item.factor}`,
      project: mutateProject(project, item.label, (node) =>
        node.kind === "multi_branch_residual" && typeof node.parameters.drop_path === "number"
          ? { ...node, parameters: { ...node.parameters, drop_path: Number((node.parameters.drop_path * item.factor).toFixed(3)) } }
          : node,
      ),
    }));
  }

  if (preset === "pooling") {
    return [
      { id: "pool-gap", label: "GAP", mode: "gap" },
      { id: "pool-gap-gmp", label: "GAP + GMP", mode: "gap_gmp" },
    ].map((item) => ({
      id: item.id,
      label: item.label,
      note: "Pooling mode",
      project: mutateProject(project, item.label, (node) =>
        node.kind === "pooling_fusion" ? { ...node, parameters: { ...node.parameters, mode: item.mode } } : node,
      ),
    }));
  }

  return [
    { id: "se-off", label: "SE off", useSe: false },
    { id: "se-on", label: "SE on", useSe: true },
  ].map((item) => ({
    id: item.id,
    label: item.label,
    note: "Squeeze-excite toggle",
    project: mutateProject(project, item.label, (node) =>
      node.kind === "multi_branch_residual" ? { ...node, parameters: { ...node.parameters, use_se: item.useSe } } : node,
    ),
  }));
}

function nearestMultiple(value: number, multiple: number) {
  return Math.max(multiple, Math.round(value / multiple) * multiple);
}

function mutateProject(project: TopologyProject, suffix: string, mutateNode: (node: TopologyNode) => TopologyNode): TopologyProject {
  return {
    ...project,
    name: `${project.name} / ${suffix}`,
    nodes: project.nodes.map((node) => mutateNode({ ...node, parameters: { ...node.parameters }, position: { ...node.position } })),
    edges: project.edges.map((edge) => ({ ...edge })),
  };
}

function updateQueueItem(queue: AblationQueueItem[], id: string, patch: Partial<AblationQueueItem>) {
  return queue.map((item) => (item.id === id ? { ...item, ...patch } : item));
}

function buildChartRuns(runHistory: GeneratedRunSummary[], currentMetrics: GeneratedTrainMetrics, queue: AblationQueueItem[]) {
  const queueResults = queue
    .filter((item) => item.metrics)
    .map((item) => ({
      label: item.label,
      accuracy: item.metrics?.best_accuracy ?? item.metrics?.test_accuracy ?? undefined,
      trainLoss: item.metrics?.train_loss ?? undefined,
      testLoss: item.metrics?.test_loss ?? undefined,
    }));
  if (queueResults.length > 0) {
    return queueResults;
  }

  const completed = runHistory
    .filter((run) => run.metrics && (run.metrics.test_accuracy !== null || run.metrics.train_loss !== null || run.metrics.test_loss !== null))
    .slice(0, 11)
    .reverse()
    .map((run) => ({
      label: formatRunLabel(run.runId),
      accuracy: run.metrics.best_accuracy ?? run.metrics.test_accuracy ?? undefined,
      trainLoss: run.metrics.train_loss ?? undefined,
      testLoss: run.metrics.test_loss ?? undefined,
    }));
  if (currentMetrics?.status === "running") {
    completed.push({
      label: "Live",
      accuracy: currentMetrics.best_accuracy ?? currentMetrics.test_accuracy ?? undefined,
      trainLoss: currentMetrics.train_loss ?? undefined,
      testLoss: currentMetrics.test_loss ?? undefined,
    });
  }
  return completed;
}

function buildQueueReportRows(queue: AblationQueueItem[]) {
  return queue.map((item) => {
    const resolution = resolveTopology(item.project);
    return {
      label: item.label,
      note: item.note,
      topologyId: item.topologyId,
      status: item.status,
      runId: item.runId ?? "",
      runPath: item.runId ? `runs/${item.runId}` : "",
      accuracy: item.metrics?.test_accuracy ?? null,
      baselineAccuracy: item.metrics?.baseline_accuracy ?? null,
      bestAccuracy: item.metrics?.best_accuracy ?? item.metrics?.test_accuracy ?? null,
      bestEpoch: item.metrics?.best_epoch ?? null,
      accuracyDelta: item.metrics?.accuracy_delta ?? null,
      trainAccuracy: item.metrics?.train_accuracy ?? null,
      trainLoss: item.metrics?.train_loss ?? null,
      testLoss: item.metrics?.test_loss ?? null,
      finalBatchLoss: item.metrics?.final_batch_loss ?? null,
      parameters: resolution.totalParameters,
      flops: resolution.totalFlops,
      projectName: item.project.name,
    };
  });
}

function bestQueueVariant(queue: AblationQueueItem[]) {
  return queue
    .filter((item) => typeof (item.metrics?.best_accuracy ?? item.metrics?.test_accuracy) === "number")
    .sort((left, right) => (right.metrics?.best_accuracy ?? right.metrics?.test_accuracy ?? -1) - (left.metrics?.best_accuracy ?? left.metrics?.test_accuracy ?? -1))[0];
}

function hydrateExperimentQueue(value: unknown): AblationQueueItem[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item): AblationQueueItem | null => {
      if (!isRecord(item)) {
        return null;
      }
      const parsed = parseTopologyProject(item.project);
      if (!parsed.project) {
        return null;
      }
      const id = typeof item.id === "string" ? item.id : topologyVersionId(parsed.project);
      const status = isQueueStatus(item.status) ? item.status : "queued";
      return {
        id,
        label: typeof item.label === "string" ? item.label : id,
        note: typeof item.note === "string" ? item.note : "",
        project: parsed.project,
        topologyId: typeof item.topologyId === "string" ? item.topologyId : topologyVersionId(parsed.project),
        status,
        runId: typeof item.runId === "string" ? item.runId : undefined,
        metrics: isRecord(item.metrics) ? (item.metrics as NonNullable<GeneratedTrainMetrics>) : undefined,
        error: typeof item.error === "string" ? item.error : undefined,
      };
    })
    .filter((item): item is AblationQueueItem => Boolean(item));
}

function isQueuePreset(value: unknown): value is QueuePreset {
  return value === "width" || value === "dropPath" || value === "pooling" || value === "se";
}

function isQueueStatus(value: unknown): value is AblationQueueItem["status"] {
  return value === "queued" || value === "running" || value === "complete" || value === "failed" || value === "cancelled";
}

function defaultExperimentName(preset: QueuePreset) {
  const prefix =
    preset === "width" ? "width-sweep" : preset === "dropPath" ? "drop-path-sweep" : preset === "pooling" ? "pooling-sweep" : "se-sweep";
  return `${prefix}-001`;
}

function toCsv(rows: ReturnType<typeof buildQueueReportRows>) {
  const headers = [
    "label",
    "note",
    "topologyId",
    "status",
    "runId",
    "runPath",
    "accuracy",
    "baselineAccuracy",
    "bestAccuracy",
    "bestEpoch",
    "accuracyDelta",
    "trainAccuracy",
    "trainLoss",
    "testLoss",
    "finalBatchLoss",
    "parameters",
    "flops",
    "projectName",
  ];
  return [headers.join(","), ...rows.map((row) => headers.map((header) => csvCell(row[header as keyof typeof row])).join(","))].join("\n");
}

function csvCell(value: string | number | null) {
  const text = value === null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadTextFile(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseBatchLosses(logs: string[]) {
  return logs
    .map((line) => /batch \d+\/\d+ loss=([0-9.]+)/.exec(line)?.[1])
    .filter((value): value is string => Boolean(value))
    .map(Number);
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
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

function formatMaybePercent(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? formatPercent(value) : "-";
}

function formatMaybeSignedPercent(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  const formatted = formatPercent(value);
  return value > 0 ? `+${formatted}` : formatted;
}

function formatMaybeNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

function formatRunLabel(runId: string) {
  const [timestamp] = runId.split("_");
  if (!timestamp || timestamp.length < 15) {
    return runId;
  }
  return `${timestamp.slice(4, 6)}/${timestamp.slice(6, 8)} ${timestamp.slice(9, 11)}:${timestamp.slice(11, 13)}`;
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
