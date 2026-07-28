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
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent, type ReactNode } from "react";

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
  preset: RunPreset;
  epochs: number;
  trainLimit: number;
  testLimit: number;
  batchSize: number;
  learningRate: number;
  seed: number;
  cpu: boolean;
};
type RunPreset = "fast" | "balanced" | "stronger" | "custom";
type QueuePreset = "width" | "dropPath" | "pooling" | "se";
type VariantScope = "selected" | "all";
type VariantBuilderSettings = {
  scope: VariantScope;
  widthDelta: number;
  dropPathDelta: number;
  poolingMode: "gap" | "gap_gmp";
  useSe: boolean;
};
type GeneratedTrainMetrics = {
  status: string;
  run_id?: string;
  topology_id?: string;
  created_at?: string;
  updated_at?: string;
  epochs: number;
  train_limit: number;
  test_limit: number;
  batch_size?: number;
  seed?: number;
  learning_rate?: number;
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
  diagnostics_path?: string;
  diagnostics?: RunDiagnostics | null;
  replay_comparison?: NonNullable<ReplayComparison> | null;
  checkpoint?: string;
  duration_seconds?: number;
  passed_smoke_rule: boolean | null;
  error?: string;
} | null;
type RunDiagnostics = {
  created_at?: string;
  test_loss?: number;
  test_samples?: number;
  confusion_matrix?: number[][];
  per_class_accuracy?: Array<{ digit: number; accuracy: number | null; correct: number; total: number }>;
  prediction_samples?: PredictionSample[];
};
type PredictionSample = {
  truth: number;
  predicted: number;
  correct: boolean;
  confidence?: number;
  pixels: number[];
};
type TopologyDiff = {
  lines: string[];
  summaryText: string;
  paramsDelta: number;
  flopsDelta: number;
  changedFieldCount: number;
  addedNodes: string[];
  removedNodes: string[];
  addedEdges: string[];
  removedEdges: string[];
  totalChanges: number;
};
type WinnerSummary = {
  title: string;
  detail: string;
  diffSummary: string;
  bestId: string;
};
type VariantPreview = {
  project: TopologyProject;
  label: string;
  note: string;
  diff: TopologyDiff;
  targetLabel: string;
};
type ReplayComparison = {
  createdAt: string;
  savedAt?: string;
  originalRunId: string;
  replayRunId: string;
  originalAccuracy: number | null;
  replayAccuracy: number | null;
  originalLoss: number | null;
  replayLoss: number | null;
  accuracyDelta: number | null;
  lossDelta: number | null;
  reproducible: boolean;
  tolerance: {
    accuracy: number;
    loss: number;
  };
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
type ArchitectureSummary = {
  id: string;
  name: string;
  notes: string;
  tags: string[];
  parentId: string;
  topologyId: string;
  params: number;
  flops: number;
  createdAt: string;
  updatedAt: string;
  archived: boolean;
  locked: boolean;
  project: TopologyProject;
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

const RUN_PRESETS: Record<Exclude<RunPreset, "custom">, Omit<TrainSettings, "preset" | "cpu">> = {
  fast: { epochs: 1, trainLimit: 1024, testLimit: 512, batchSize: 128, learningRate: 0.001, seed: 7 },
  balanced: { epochs: 2, trainLimit: 2048, testLimit: 1024, batchSize: 128, learningRate: 0.001, seed: 7 },
  stronger: { epochs: 3, trainLimit: 4096, testLimit: 2048, batchSize: 128, learningRate: 0.0008, seed: 7 },
};

const RUN_PRESET_LABELS: Record<RunPreset, string> = {
  fast: "Fast",
  balanced: "Balanced",
  stronger: "Stronger",
  custom: "Custom",
};
const REPLAY_TOLERANCE = {
  accuracy: 0.005,
  loss: 0.25,
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
  const [replayComparison, setReplayComparison] = useState<ReplayComparison>(null);
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
  const [architectures, setArchitectures] = useState<ArchitectureSummary[]>([]);
  const [selectedArchitectureId, setSelectedArchitectureId] = useState("baseline-fiveblock");
  const [architectureBusy, setArchitectureBusy] = useState(false);
  const [trainSettings, setTrainSettings] = useState<TrainSettings>({
    preset: "balanced",
    ...RUN_PRESETS.balanced,
    cpu: true,
  });
  const [variantBuilder, setVariantBuilder] = useState<VariantBuilderSettings>({
    scope: "selected",
    widthDelta: 0,
    dropPathDelta: 0,
    poolingMode: "gap_gmp",
    useSe: true,
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
  const selectedArchitecture = architectures.find((architecture) => architecture.id === selectedArchitectureId);
  const canvasMatchesSelectedArchitecture = selectedArchitecture ? selectedArchitecture.topologyId === currentTopologyId : false;
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

  const loadArchitectures = useCallback(async () => {
    setArchitectureBusy(true);
    try {
      const response = await fetch("/api/architectures", { cache: "no-store" });
      const payload = await response.json();
      const nextArchitectures = Array.isArray(payload.architectures) ? payload.architectures : [];
      setArchitectures(nextArchitectures);
      if (!nextArchitectures.some((architecture: ArchitectureSummary) => architecture.id === selectedArchitectureId)) {
        setSelectedArchitectureId(nextArchitectures[0]?.id ?? "baseline-fiveblock");
      }
    } finally {
      setArchitectureBusy(false);
    }
  }, [selectedArchitectureId]);

  useEffect(() => {
    void loadGeneratedMetrics();
    void loadGeneratedRuns();
    void loadExperiments();
    void loadArchitectures();
  }, [loadArchitectures, loadExperiments, loadGeneratedMetrics, loadGeneratedRuns]);

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

  async function startGeneratedRun(targetProject: TopologyProject, settingsOverride: TrainSettings = trainSettings) {
    const response = await fetch("/api/train-generated", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project: targetProject, ...settingsOverride }),
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

  async function trainVariantProject(variantProject: TopologyProject, label: string, note: string) {
    const variantId = `builder-${Date.now()}`;
    const variantTopologyId = topologyVersionId(variantProject);
    const baselineMetrics =
      trainMetrics?.topology_id === currentTopologyId || (!trainMetrics?.topology_id && trainMetrics) ? (trainMetrics as NonNullable<GeneratedTrainMetrics>) : undefined;
    const baselineItem: AblationQueueItem = {
      id: "builder-baseline",
      label: "Baseline",
      note: "Current canvas before variant",
      project,
      topologyId: currentTopologyId,
      status: baselineMetrics ? "complete" : "queued",
      metrics: baselineMetrics,
    };
    const variantItem: AblationQueueItem = {
      id: variantId,
      label,
      note,
      project: variantProject,
      topologyId: variantTopologyId,
      status: "running",
    };
    setAblationQueue((current) => {
      const withoutOldBuilder = current.filter((item) => !item.id.startsWith("builder-"));
      return [baselineItem, ...withoutOldBuilder, variantItem];
    });
    setTrainingBusy(true);
    setTrainingLogs([]);
    setNotice({ tone: "good", text: `Training ${label}.` });
    try {
      const started = await startGeneratedRun(variantProject);
      setAblationQueue((current) => updateQueueItem(current, variantId, { runId: started.runId }));
      const finished = await waitForGeneratedRun(started.runId);
      setAblationQueue((current) =>
        updateQueueItem(current, variantId, {
          status: finished.status,
          metrics: finished.metrics,
          error: finished.error,
        }),
      );
      setNotice({ tone: finished.status === "complete" ? "good" : "bad", text: finished.status === "complete" ? `${label} completed.` : `${label} did not pass.` });
    } catch (error) {
      setAblationQueue((current) => updateQueueItem(current, variantId, { status: "failed", error: error instanceof Error ? error.message : "Variant training failed." }));
      setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Variant training failed." });
    } finally {
      setTrainingBusy(false);
      setActiveRunId(null);
      void loadGeneratedMetrics();
      void loadGeneratedRuns();
    }
  }

  function promoteVariantProject(variantProject: TopologyProject, label: string) {
    commitProject(variantProject);
    const nextSelected = variantProject.nodes.some((node) => node.id === selectedNodeId) ? selectedNodeId : variantProject.nodes[0]?.id;
    if (nextSelected) {
      setSelectedNodeId(nextSelected);
    }
    setNotice({ tone: "good", text: `${label} promoted to the canvas.` });
  }

  async function loadArchitecture(id: string) {
    const architecture = architectures.find((item) => item.id === id);
    if (!architecture) {
      setNotice({ tone: "bad", text: "Architecture not found in the library." });
      return;
    }
    setSelectedArchitectureId(id);
    setHistory({ past: [], present: architecture.project, future: [] });
    setSelectedNodeId(architecture.project.nodes.some((node) => node.id === selectedNodeId) ? selectedNodeId : architecture.project.nodes[0]?.id ?? "input");
    setAblationQueue([]);
    setSelectedRunId(null);
    setNotice({ tone: "good", text: `Loaded ${architecture.name}.` });
  }

  async function saveCurrentArchitecture() {
    const name = window.prompt("Architecture name", project.name);
    if (!name?.trim()) {
      return;
    }
    const notes = window.prompt("Notes", "") ?? "";
    const response = await fetch("/api/architectures", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        notes,
        tags: ["saved"],
        parentId: selectedArchitectureId,
        project: { ...project, name },
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      setNotice({ tone: "bad", text: payload?.error ?? "Could not save architecture." });
      return;
    }
    await loadArchitectures();
    setSelectedArchitectureId(payload.architecture.id);
    setHistory({ past: [], present: payload.architecture.project, future: [] });
    setAblationQueue([]);
    setSelectedRunId(null);
    setNotice({ tone: "good", text: `Saved architecture ${payload.architecture.name}.` });
  }

  async function duplicateArchitecture() {
    const current = architectures.find((item) => item.id === selectedArchitectureId);
    const name = window.prompt("Duplicate architecture as", `${project.name} copy`);
    if (!name?.trim()) {
      return;
    }
    const response = await fetch("/api/architectures", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        notes: current?.notes ? `Duplicate of ${current.name}. ${current.notes}` : `Duplicate of ${current?.name ?? project.name}.`,
        tags: [...(current?.tags ?? []), "duplicate"],
        parentId: current?.id ?? selectedArchitectureId,
        sourceId: current?.id ?? selectedArchitectureId,
        project: { ...project, name },
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      setNotice({ tone: "bad", text: payload?.error ?? "Could not duplicate architecture." });
      return;
    }
    await loadArchitectures();
    setSelectedArchitectureId(payload.architecture.id);
    setHistory({ past: [], present: payload.architecture.project, future: [] });
    setAblationQueue([]);
    setSelectedRunId(null);
    setNotice({ tone: "good", text: `Duplicated as ${payload.architecture.name}.` });
  }

  async function archiveArchitecture() {
    const current = architectures.find((item) => item.id === selectedArchitectureId);
    if (!current) {
      return;
    }
    if (current.locked) {
      setNotice({ tone: "bad", text: "Built-in architectures cannot be archived. Duplicate it first if you want your own editable copy." });
      return;
    }
    const confirmed = window.confirm(`Archive ${current.name}?`);
    if (!confirmed) {
      return;
    }
    const response = await fetch(`/api/architectures?id=${encodeURIComponent(selectedArchitectureId)}`, { method: "DELETE" });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      setNotice({ tone: "bad", text: payload?.error ?? "Could not archive architecture." });
      return;
    }
    await loadArchitectures();
    const baseline = architectures.find((item) => item.id === "baseline-fiveblock");
    if (baseline) {
      setHistory({ past: [], present: baseline.project, future: [] });
    }
    setSelectedArchitectureId("baseline-fiveblock");
    setNotice({ tone: "good", text: `${current.name} archived.` });
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
    const winnerSummary = buildWinnerSummary(ablationQueue);
    if (rows.length === 0) {
      setNotice({ tone: "bad", text: "No queue variants to export yet." });
      return;
    }
    if (format === "json") {
      downloadTextFile("mnist-ablation-report.json", JSON.stringify({ exportedAt: new Date().toISOString(), winnerSummary, rows }, null, 2), "application/json");
      setNotice({ tone: "good", text: "JSON comparison report exported." });
      return;
    }
    downloadTextFile("mnist-ablation-report.csv", toCsv(rows), "text/csv");
    setNotice({ tone: "good", text: "CSV comparison report exported." });
  }

  function exportRunBundle() {
    const selectedRun = runHistory.find((run) => run.runId === selectedRunId);
    const metrics = selectedRun?.metrics ?? trainMetrics;
    const bundle = buildRunBundle({
      project,
      metrics,
      settings: trainSettings,
      resolution,
      diff: buildTopologyDiff(enhancedFiveBlockTopology, project),
      generatedFiles,
      queue: ablationQueue,
      replayComparisons: replayComparisonsFromRuns(runHistory, replayComparison),
    });
    downloadTextFile(`mnist-run-bundle-${currentTopologyId}.json`, JSON.stringify(bundle, null, 2), "application/json");
    setNotice({ tone: "good", text: "Run bundle exported." });
  }

  async function replayCurrentRun() {
    const selectedRun = runHistory.find((run) => run.runId === selectedRunId);
    const originalMetrics = selectedRun?.metrics ?? trainMetrics;
    if (!originalMetrics) {
      setNotice({ tone: "bad", text: "No original run metrics to replay yet." });
      return;
    }
    if (originalMetrics.topology_id && originalMetrics.topology_id !== currentTopologyId) {
      setNotice({ tone: "bad", text: "Selected run topology differs from the canvas. Load that topology before replaying." });
      return;
    }
    await replayProjectRun(project, originalMetrics, selectedRun?.runId ?? originalMetrics.run_id ?? "latest");
  }

  async function replayBestVariant() {
    const best = bestQueueVariant(ablationQueue);
    if (!best?.metrics) {
      setNotice({ tone: "bad", text: "No completed queue winner to replay yet." });
      return;
    }
    await replayProjectRun(best.project, best.metrics, best.runId ?? best.metrics.run_id ?? best.id);
  }

  async function replayProjectRun(targetProject: TopologyProject, originalMetrics: NonNullable<GeneratedTrainMetrics>, originalRunId: string) {
    const replaySettings = settingsFromMetrics(originalMetrics, trainSettings);
    setTrainingBusy(true);
    setTrainingLogs([]);
    setReplayComparison(null);
    setNotice({ tone: "good", text: "Replay run started." });
    try {
      const started = await startGeneratedRun(targetProject, replaySettings);
      const finished = await waitForGeneratedRun(started.runId);
      const comparison = compareReplay(originalRunId, started.runId, originalMetrics, finished.metrics ?? null);
      setReplayComparison(comparison);
      await saveReplayComparison(comparison);
      setNotice({ tone: comparison.reproducible ? "good" : "bad", text: comparison.reproducible ? "Replay matches original within tolerance." : "Replay finished outside tolerance." });
      void loadGeneratedRuns();
    } catch (error) {
      setNotice({ tone: "bad", text: error instanceof Error ? error.message : "Replay run failed." });
    } finally {
      setTrainingBusy(false);
      setActiveRunId(null);
      void loadGeneratedMetrics();
    }
  }

  async function saveReplayComparison(comparison: NonNullable<ReplayComparison>) {
    const response = await fetch("/api/replay-comparison", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ replayRunId: comparison.replayRunId, comparison }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.error ?? "Could not save replay comparison.");
    }
    if (payload?.comparison) {
      setReplayComparison(payload.comparison);
    }
  }

  async function lockWinner() {
    const best = bestQueueVariant(ablationQueue);
    if (!best?.metrics) {
      setNotice({ tone: "bad", text: "No completed queue winner to lock yet." });
      return;
    }
    const name = experimentName.trim() || defaultExperimentName(queuePreset);
    const reportRows = buildQueueReportRows(ablationQueue);
    const winnerSummary = buildWinnerSummary(ablationQueue);
    const response = await fetch("/api/experiments/winner", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        experimentName: name,
        variantId: best.id,
        label: best.label,
        project: best.project,
        metrics: best.metrics,
        diagnostics: best.metrics.diagnostics ?? null,
        replayComparisons: replayComparisonsFromRuns(runHistory, replayComparison),
        report: { winnerSummary, rows: reportRows },
        manifest: {
          topologyDiff: buildTopologyDiff(enhancedFiveBlockTopology, best.project),
          trainSettings: settingsFromMetrics(best.metrics, trainSettings),
        },
      }),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      setNotice({ tone: "bad", text: payload?.error ?? "Could not lock winner." });
      return;
    }
    setNotice({ tone: "good", text: `Winner locked at ${payload.winner?.directory ?? `experiments/${name}/winner`}.` });
    await loadExperiments();
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
        winnerSummary: buildWinnerSummary(queue),
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
        <div className="architectureTopbar">
          <label>
            <span>Architecture</span>
            <select value={selectedArchitectureId} onChange={(event) => void loadArchitecture(event.target.value)} disabled={architectureBusy}>
              {architectures.map((architecture) => (
                <option key={architecture.id} value={architecture.id}>
                  {architecture.name}
                </option>
              ))}
            </select>
          </label>
          <span className="topologyIdPill" title={currentTopologyId}>
            ID {currentTopologyId}
          </span>
          <button type="button" onClick={() => void saveCurrentArchitecture()} disabled={architectureBusy} title="Save current topology as a library architecture">
            <Save size={16} />
            Save new
          </button>
          <button type="button" onClick={() => void duplicateArchitecture()} disabled={architectureBusy} title="Duplicate selected architecture">
            <Plus size={16} />
            Duplicate
          </button>
          <button
            type="button"
            onClick={() => void archiveArchitecture()}
            disabled={architectureBusy || Boolean(architectures.find((architecture) => architecture.id === selectedArchitectureId)?.locked)}
            title="Archive selected saved architecture"
          >
            <Trash2 size={16} />
            Archive
          </button>
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

      <ArchitectureLibraryPanel
        architectures={architectures}
        selectedId={selectedArchitectureId}
        runHistory={runHistory}
        currentTopologyId={currentTopologyId}
        busy={architectureBusy}
        onLoad={(id) => void loadArchitecture(id)}
        onSave={() => void saveCurrentArchitecture()}
        onDuplicate={() => void duplicateArchitecture()}
        onArchive={() => void archiveArchitecture()}
      />

      <GeneratedTrainingPanel
        project={project}
        metrics={trainMetrics}
        metricsBusy={metricsBusy}
        trainingBusy={trainingBusy}
        activeRunId={activeRunId}
        logs={trainingLogs}
        replayComparison={replayComparison}
        runHistory={runHistory}
        runsBusy={runsBusy}
        selectedRunId={selectedRunId}
        currentTopologyId={currentTopologyId}
        selectedNodeId={selectedNodeId}
        variantBuilder={variantBuilder}
        ablationQueue={ablationQueue}
        queueBusy={queueBusy}
        queuePreset={queuePreset}
        experimentName={experimentName}
        savedExperiments={savedExperiments}
        selectedExperimentName={selectedExperimentName}
        experimentsBusy={experimentsBusy}
        settings={trainSettings}
        onSettingsChange={setTrainSettings}
        onVariantBuilderChange={setVariantBuilder}
        onTrainVariant={(variantProject, label, note) => void trainVariantProject(variantProject, label, note)}
        onPromoteVariant={promoteVariantProject}
        onExperimentNameChange={setExperimentName}
        onQueuePresetChange={changeQueuePreset}
        onLoadExperiment={loadSavedExperiment}
        onSaveExperiment={() => void saveExperimentSnapshot()}
        onLoadBest={loadBestVariant}
        onExportBundle={exportRunBundle}
        onReplayRun={replayCurrentRun}
        onReplayBest={replayBestVariant}
        onLockWinner={() => void lockWinner()}
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
            <strong>Current architecture</strong>
            <BrainCircuit size={18} />
          </div>
          <div className={`templateCard active ${canvasMatchesSelectedArchitecture ? "" : "dirty"}`}>
            <strong>{project.name}</strong>
            <span>{topologySummary(project, resolution)}</span>
            <div className="architectureMiniGrid">
              <span>
                Topology
                <strong>{currentTopologyId}</strong>
              </span>
              <span>
                Source
                <strong>{selectedArchitecture?.locked ? "built-in" : selectedArchitecture ? "saved" : "canvas"}</strong>
              </span>
              <span>
                Params
                <strong>{formatCompactNumber(resolution.totalParameters)}</strong>
              </span>
              <span>
                Status
                <strong>{canvasMatchesSelectedArchitecture ? "matched" : "edited"}</strong>
              </span>
            </div>
            {selectedArchitecture?.notes ? <small>{selectedArchitecture.notes}</small> : null}
          </div>
          <button className="restoreButton" type="button" onClick={restoreTemplate}>
            <RefreshCw size={15} />
            Load baseline
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

function ArchitectureLibraryPanel({
  architectures,
  selectedId,
  runHistory,
  currentTopologyId,
  busy,
  onLoad,
  onSave,
  onDuplicate,
  onArchive,
}: {
  architectures: ArchitectureSummary[];
  selectedId: string;
  runHistory: GeneratedRunSummary[];
  currentTopologyId: string;
  busy: boolean;
  onLoad: (id: string) => void;
  onSave: () => void;
  onDuplicate: () => void;
  onArchive: () => void;
}) {
  const selectedArchitecture = architectures.find((architecture) => architecture.id === selectedId);
  const canvasMatchesSelected = selectedArchitecture ? selectedArchitecture.topologyId === currentTopologyId : true;

  return (
    <section className="architectureLibraryPanel" aria-label="Architecture library">
      <div className="architectureLibraryHeader">
        <div>
          <p className="eyebrow">Architecture library</p>
          <h2>Saved working architectures</h2>
        </div>
        <div className="architectureLibraryActions">
          <button type="button" onClick={onSave} disabled={busy}>
            <Save size={16} />
            Save current
          </button>
          <button type="button" onClick={onDuplicate} disabled={busy}>
            <Plus size={16} />
            Duplicate
          </button>
          <button type="button" onClick={onArchive} disabled={busy || Boolean(selectedArchitecture?.locked)}>
            <Trash2 size={16} />
            Archive
          </button>
        </div>
      </div>
      <div className="architectureHomeGrid">
        <div>
          <span>Current architecture</span>
          <strong>{selectedArchitecture?.name ?? "Unsaved canvas"}</strong>
          <small>{canvasMatchesSelected ? "canvas matches library topology" : "canvas has unsaved topology changes"}</small>
        </div>
        <div>
          <span>Topology ID</span>
          <strong>{currentTopologyId}</strong>
          <small>{selectedArchitecture?.parentId ? `parent ${selectedArchitecture.parentId}` : "baseline parent"}</small>
        </div>
        <div>
          <span>Best run</span>
          <strong>{formatMaybePercent(architectureRunStats(selectedArchitecture, runHistory).bestAccuracy)}</strong>
          <small>{architectureRunStats(selectedArchitecture, runHistory).bestRunId ? formatRunLabel(architectureRunStats(selectedArchitecture, runHistory).bestRunId ?? "") : "no saved run yet"}</small>
        </div>
        <div>
          <span>Latest replay</span>
          <strong>{latestReplayStatusForTopology(currentTopologyId, runHistory)}</strong>
          <small>from saved run history</small>
        </div>
      </div>
      <div className="architectureLibraryGrid">
        {architectures.map((architecture) => {
          const stats = architectureRunStats(architecture, runHistory);
          const selected = architecture.id === selectedId;
          return (
            <button className={`architectureCard ${selected ? "selected" : ""}`} key={architecture.id} type="button" onClick={() => onLoad(architecture.id)} disabled={busy}>
              <span className="architectureCardTitle">
                <strong>{architecture.name}</strong>
                {selected ? <CheckCircle2 size={17} /> : <FolderOpen size={17} />}
              </span>
              <span className="architectureCardNotes">{architecture.notes || "No notes yet."}</span>
              <span className="architectureTagRow">
                {architecture.tags.slice(0, 3).map((tag) => (
                  <small key={tag}>{tag}</small>
                ))}
                {architecture.locked ? <small>built-in</small> : <small>saved</small>}
              </span>
              <span className="architectureStats">
                <span>
                  Params
                  <strong>{formatCompactNumber(architecture.params)}</strong>
                </span>
                <span>
                  FLOPs
                  <strong>{formatCompactNumber(architecture.flops)}</strong>
                </span>
                <span>
                  Best
                  <strong>{formatMaybePercent(stats.bestAccuracy)}</strong>
                </span>
                <span>
                  Last trained
                  <strong>{stats.lastTrained ? formatShortDate(stats.lastTrained) : "-"}</strong>
                </span>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function GeneratedTrainingPanel({
  project,
  metrics,
  metricsBusy,
  trainingBusy,
  activeRunId,
  logs,
  replayComparison,
  runHistory,
  runsBusy,
  selectedRunId,
  currentTopologyId,
  selectedNodeId,
  variantBuilder,
  ablationQueue,
  queueBusy,
  queuePreset,
  experimentName,
  savedExperiments,
  selectedExperimentName,
  experimentsBusy,
  settings,
  onSettingsChange,
  onVariantBuilderChange,
  onTrainVariant,
  onPromoteVariant,
  onExperimentNameChange,
  onQueuePresetChange,
  onLoadExperiment,
  onSaveExperiment,
  onLoadBest,
  onExportBundle,
  onReplayRun,
  onReplayBest,
  onLockWinner,
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
  project: TopologyProject;
  metrics: GeneratedTrainMetrics;
  metricsBusy: boolean;
  trainingBusy: boolean;
  activeRunId: string | null;
  logs: string[];
  replayComparison: ReplayComparison;
  runHistory: GeneratedRunSummary[];
  runsBusy: boolean;
  selectedRunId: string | null;
  currentTopologyId: string;
  selectedNodeId: string;
  variantBuilder: VariantBuilderSettings;
  ablationQueue: AblationQueueItem[];
  queueBusy: boolean;
  queuePreset: QueuePreset;
  experimentName: string;
  savedExperiments: ExperimentSummary[];
  selectedExperimentName: string;
  experimentsBusy: boolean;
  settings: TrainSettings;
  onSettingsChange: (settings: TrainSettings) => void;
  onVariantBuilderChange: (settings: VariantBuilderSettings) => void;
  onTrainVariant: (variantProject: TopologyProject, label: string, note: string) => void;
  onPromoteVariant: (variantProject: TopologyProject, label: string) => void;
  onExperimentNameChange: (name: string) => void;
  onQueuePresetChange: (preset: QueuePreset) => void;
  onLoadExperiment: (name: string) => void;
  onSaveExperiment: () => void;
  onLoadBest: () => void;
  onExportBundle: () => void;
  onReplayRun: () => void;
  onReplayBest: () => void;
  onLockWinner: () => void;
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
    onSettingsChange({ ...settings, preset: key === "preset" ? (value as RunPreset) : "custom", [key]: value });
  }

  function applyRunPreset(preset: Exclude<RunPreset, "custom">) {
    onSettingsChange({ preset, ...RUN_PRESETS[preset], cpu: settings.cpu });
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
  const improvementCheck = getImprovementCheck(displayedMetrics);
  const topologyDiff = useMemo(() => buildTopologyDiff(enhancedFiveBlockTopology, project), [project]);
  const currentResolution = useMemo(() => resolveTopology(project), [project]);
  const winnerSummary = useMemo(() => buildWinnerSummary(ablationQueue), [ablationQueue]);
  const variantPreview = useMemo(() => buildVariantPreview(project, selectedNodeId, variantBuilder), [project, selectedNodeId, variantBuilder]);

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
          <button type="button" onClick={onReplayRun} disabled={trainingBusy || queueBusy || !displayedMetrics}>
            <RefreshCw size={16} />
            Replay
          </button>
          <button type="button" onClick={onExportBundle} disabled={!displayedMetrics}>
            <Download size={16} />
            Bundle
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
          <span>Run preset</span>
          <select
            value={settings.preset}
            onChange={(event) => {
              if (event.target.value !== "custom") {
                applyRunPreset(event.target.value as Exclude<RunPreset, "custom">);
              }
            }}
          >
            {settings.preset === "custom" ? (
              <option value="custom" disabled>
                Custom
              </option>
            ) : null}
            {(["fast", "balanced", "stronger"] as const).map((preset) => (
              <option key={preset} value={preset}>
                {RUN_PRESET_LABELS[preset]}
              </option>
            ))}
          </select>
        </label>
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
        <label>
          <span>Learning rate</span>
          <input
            min={0.00001}
            max={0.1}
            step={0.0001}
            type="number"
            value={settings.learningRate}
            onChange={(event) => updateSetting("learningRate", Number(event.target.value))}
          />
        </label>
        <label>
          <span>Seed</span>
          <input
            min={0}
            max={999999}
            step={1}
            type="number"
            value={settings.seed}
            onChange={(event) => updateSetting("seed", Number(event.target.value))}
          />
        </label>
        <label className="trainingToggle">
          <span>CPU</span>
          <input type="checkbox" checked={settings.cpu} onChange={(event) => updateSetting("cpu", event.target.checked)} />
        </label>
      </div>
      <p className="trainingNote">
        Fast checks compilation quickly. Balanced is the default comparison run. Stronger spends more time for ablation decisions.
      </p>
      {improvementCheck ? (
        <div className={`improvementBadge ${improvementCheck.tone}`}>
          {improvementCheck.tone === "good" ? <CheckCircle2 size={18} /> : <TriangleAlert size={18} />}
          <div>
            <strong>{improvementCheck.title}</strong>
            <span>{improvementCheck.detail}</span>
          </div>
        </div>
      ) : null}
      <VariantBuilderPanel
        settings={variantBuilder}
        preview={variantPreview}
        trainingBusy={trainingBusy || queueBusy}
        onSettingsChange={onVariantBuilderChange}
        onTrainVariant={() => onTrainVariant(variantPreview.project, variantPreview.label, variantPreview.note)}
        onPromoteVariant={() => onPromoteVariant(variantPreview.project, variantPreview.label)}
      />
      <TraceabilityPanel
        topologyId={currentTopologyId}
        metrics={displayedMetrics}
        settings={settings}
        resolution={currentResolution}
        diff={topologyDiff}
        winnerSummary={winnerSummary}
        replayComparison={replayComparison}
        onReplayBest={onReplayBest}
        onLockWinner={onLockWinner}
      />
      <ReplayHistoryPanel comparisons={replayComparisonsFromRuns(runHistory, replayComparison)} />
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
            LR / seed <strong>{formatLearningRate(displayedMetrics.learning_rate)} / {displayedMetrics.seed ?? "-"}</strong>
          </span>
          <span>
            Checkpoint <strong>{displayedMetrics.checkpoint ?? "-"}</strong>
          </span>
        </div>
      ) : (
        <div className="emptyTraining">No generated training metrics yet.</div>
      )}
      <MetricCharts runs={chartRuns} batchLosses={batchLosses} />
      <DiagnosticsPanel diagnostics={displayedMetrics?.diagnostics} />
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

function VariantBuilderPanel({
  settings,
  preview,
  trainingBusy,
  onSettingsChange,
  onTrainVariant,
  onPromoteVariant,
}: {
  settings: VariantBuilderSettings;
  preview: VariantPreview;
  trainingBusy: boolean;
  onSettingsChange: (settings: VariantBuilderSettings) => void;
  onTrainVariant: () => void;
  onPromoteVariant: () => void;
}) {
  function update<K extends keyof VariantBuilderSettings>(key: K, value: VariantBuilderSettings[K]) {
    onSettingsChange({ ...settings, [key]: value });
  }

  return (
    <div className="variantBuilderPanel">
      <div className="subPanelHeader">
        <strong>Variant builder</strong>
        <span>{preview.targetLabel}</span>
      </div>
      <div className="variantBuilderGrid">
        <div className="variantControls">
          <label>
            <span>Apply to</span>
            <select value={settings.scope} onChange={(event) => update("scope", event.target.value as VariantScope)}>
              <option value="selected">Selected block</option>
              <option value="all">All residual blocks</option>
            </select>
          </label>
          <label>
            <span>Width delta <strong>{formatSignedInteger(settings.widthDelta)}</strong></span>
            <input min={-64} max={64} step={4} type="range" value={settings.widthDelta} onChange={(event) => update("widthDelta", Number(event.target.value))} />
          </label>
          <label>
            <span>Drop-path delta <strong>{formatSignedDecimal(settings.dropPathDelta)}</strong></span>
            <input
              min={-0.05}
              max={0.05}
              step={0.005}
              type="range"
              value={settings.dropPathDelta}
              onChange={(event) => update("dropPathDelta", Number(event.target.value))}
            />
          </label>
          <div className="variantToggles">
            <label>
              <span>Pooling</span>
              <select value={settings.poolingMode} onChange={(event) => update("poolingMode", event.target.value as VariantBuilderSettings["poolingMode"])}>
                <option value="gap">GAP</option>
                <option value="gap_gmp">GAP + GMP</option>
              </select>
            </label>
            <label className="variantToggle">
              <span>SE</span>
              <input type="checkbox" checked={settings.useSe} onChange={(event) => update("useSe", event.target.checked)} />
            </label>
          </div>
        </div>
        <div className="variantPreview">
          <div className="diagnosticsTitle">
            <strong>Architecture change preview</strong>
            <span>{preview.label}</span>
          </div>
          <div className="variantPreviewStats">
            <span>
              Params <strong>{formatDeltaCompact(preview.diff.paramsDelta)}</strong>
            </span>
            <span>
              FLOPs <strong>{formatDeltaCompact(preview.diff.flopsDelta)}</strong>
            </span>
            <span>
              Queue label <strong>{preview.label}</strong>
            </span>
          </div>
          <ul className="diffList">
            {(preview.diff.lines.length > 0 ? preview.diff.lines : ["No changes from the current canvas."]).slice(0, 6).map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          <div className="variantActions">
            <button type="button" onClick={onTrainVariant} disabled={trainingBusy || preview.diff.totalChanges === 0}>
              <Play size={15} />
              Train this variant
            </button>
            <button type="button" onClick={onPromoteVariant} disabled={trainingBusy || preview.diff.totalChanges === 0}>
              <Upload size={15} />
              Promote to canvas
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function TraceabilityPanel({
  topologyId,
  metrics,
  settings,
  resolution,
  diff,
  winnerSummary,
  replayComparison,
  onReplayBest,
  onLockWinner,
}: {
  topologyId: string;
  metrics: GeneratedTrainMetrics;
  settings: TrainSettings;
  resolution: ReturnType<typeof resolveTopology>;
  diff: TopologyDiff;
  winnerSummary: WinnerSummary | null;
  replayComparison: ReplayComparison;
  onReplayBest: () => void;
  onLockWinner: () => void;
}) {
  return (
    <div className="traceabilityPanel">
      <div className="subPanelHeader">
        <strong>Architecture traceability</strong>
        <span>{diff.totalChanges === 0 ? "baseline topology" : `${diff.totalChanges} changes`}</span>
      </div>
      <div className="traceabilityGrid">
        <div className="manifestCard">
          <div className="diagnosticsTitle">
            <strong>Run manifest</strong>
            <span>current canvas</span>
          </div>
          <div className="manifestRows">
            <span>
              Topology <strong>{topologyId}</strong>
            </span>
            <span>
              Preset <strong>{RUN_PRESET_LABELS[settings.preset]}</strong>
            </span>
            <span>
              Seed / LR <strong>{metrics?.seed ?? settings.seed} / {formatLearningRate(metrics?.learning_rate ?? settings.learningRate)}</strong>
            </span>
            <span>
              Params <strong>{formatCompactNumber(resolution.totalParameters)} ({formatDeltaCompact(diff.paramsDelta)})</strong>
            </span>
            <span>
              FLOPs <strong>{formatCompactNumber(resolution.totalFlops)} ({formatDeltaCompact(diff.flopsDelta)})</strong>
            </span>
          </div>
        </div>
        <div className="diffCard">
          <div className="diagnosticsTitle">
            <strong>Diff against baseline</strong>
            <span>{diff.summaryText}</span>
          </div>
          <ul className="diffList">
            {(diff.lines.length > 0 ? diff.lines : ["No architecture changes from the default topology."]).slice(0, 8).map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
        <div className="winnerCard">
          <div className="diagnosticsTitle">
            <strong>Winner explanation</strong>
            <span>{winnerSummary ? "queue result" : "waiting for queue"}</span>
          </div>
          {winnerSummary ? (
            <div className="winnerSummary">
              <strong>{winnerSummary.title}</strong>
              <span>{winnerSummary.detail}</span>
              <small>{winnerSummary.diffSummary}</small>
              <div className="winnerActions">
                <button type="button" onClick={onReplayBest}>
                  <RefreshCw size={14} />
                  Replay best
                </button>
                <button type="button" onClick={onLockWinner}>
                  <Save size={14} />
                  Lock winner
                </button>
              </div>
            </div>
          ) : (
            <p className="traceabilityEmpty">Run a queue to compare variants against the current baseline.</p>
          )}
        </div>
      </div>
      {replayComparison ? (
        <div className={`replayComparison ${replayComparison.reproducible ? "good" : "bad"}`}>
          <strong>{replayComparison.reproducible ? "Replay reproducible" : "Replay drifted"}</strong>
          <span>
            Accuracy {formatMaybePercent(replayComparison.originalAccuracy)}{" -> "}{formatMaybePercent(replayComparison.replayAccuracy)}
            {" / "}loss {formatMaybeNumber(replayComparison.originalLoss)}{" -> "}{formatMaybeNumber(replayComparison.replayLoss)}
          </span>
          <small>
            {replayComparison.originalRunId} replayed as {replayComparison.replayRunId}; delta {formatMaybeSignedPercent(replayComparison.accuracyDelta)} accuracy, {formatMaybeNumber(replayComparison.lossDelta)} loss.
          </small>
        </div>
      ) : null}
    </div>
  );
}

function ReplayHistoryPanel({ comparisons }: { comparisons: NonNullable<ReplayComparison>[] }) {
  if (comparisons.length === 0) {
    return null;
  }
  return (
    <div className="replayHistoryPanel">
      <div className="subPanelHeader">
        <strong>Replay history</strong>
        <span>{comparisons.length} saved</span>
      </div>
      <div className="replayHistoryTable">
        <div className="tableHead">Original</div>
        <div className="tableHead">Replay</div>
        <div className="tableHead">Accuracy</div>
        <div className="tableHead">Loss</div>
        <div className="tableHead">Status</div>
        {comparisons.slice(0, 8).map((comparison) => (
          <Fragment key={`${comparison.originalRunId}-${comparison.replayRunId}`}>
            <span>{formatRunLabel(comparison.originalRunId)}</span>
            <span>{formatRunLabel(comparison.replayRunId)}</span>
            <strong>{formatMaybeSignedPercent(comparison.accuracyDelta)}</strong>
            <strong>{formatMaybeNumber(comparison.lossDelta)}</strong>
            <span className={comparison.reproducible ? "match" : "mismatch"}>{comparison.reproducible ? "reproducible" : "drifted"}</span>
          </Fragment>
        ))}
      </div>
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

function DiagnosticsPanel({ diagnostics }: { diagnostics?: RunDiagnostics | null }) {
  if (!diagnostics) {
    return null;
  }
  const confusion = diagnostics.confusion_matrix ?? [];
  const perClass = diagnostics.per_class_accuracy ?? [];
  const samples = diagnostics.prediction_samples ?? [];
  const maxConfusion = Math.max(...confusion.flat(), 1);
  return (
    <div className="diagnosticsPanel">
      <div className="subPanelHeader">
        <strong>Diagnostics</strong>
        <span>{diagnostics.test_samples ? `${diagnostics.test_samples.toLocaleString()} samples` : "latest run"}</span>
      </div>
      <div className="diagnosticsGrid">
        <div className="confusionPanel">
          <div className="diagnosticsTitle">
            <strong>Confusion matrix</strong>
            <span>rows true / columns predicted</span>
          </div>
          <div className="confusionMatrix" role="img" aria-label="Confusion matrix">
            <span className="axisCorner" />
            {Array.from({ length: 10 }, (_, digit) => (
              <span className="axisLabel" key={`pred-${digit}`}>
                {digit}
              </span>
            ))}
            {Array.from({ length: 10 }, (_, truth) => (
              <Fragment key={`row-${truth}`}>
                <span className="axisLabel">{truth}</span>
                {Array.from({ length: 10 }, (_, predicted) => {
                  const value = confusion[truth]?.[predicted] ?? 0;
                  const intensity = value / maxConfusion;
                  return (
                    <span
                      className={truth === predicted ? "confusionCell correct" : "confusionCell"}
                      key={`${truth}-${predicted}`}
                      style={{ "--heat": intensity } as CSSProperties}
                    >
                      {value}
                    </span>
                  );
                })}
              </Fragment>
            ))}
          </div>
        </div>
        <div className="perClassPanel">
          <div className="diagnosticsTitle">
            <strong>Per-class accuracy</strong>
            <span>digit-level signal</span>
          </div>
          <div className="perClassList">
            {Array.from({ length: 10 }, (_, digit) => {
              const item = perClass.find((entry) => entry.digit === digit);
              const accuracy = item?.accuracy;
              return (
                <div className="perClassItem" key={digit}>
                  <span>{digit}</span>
                  <div>
                    <i style={{ width: `${typeof accuracy === "number" ? Math.round(accuracy * 100) : 0}%` }} />
                  </div>
                  <strong>{formatMaybePercent(accuracy)}</strong>
                  <small>
                    {item?.correct ?? 0}/{item?.total ?? 0}
                  </small>
                </div>
              );
            })}
          </div>
        </div>
        <div className="predictionPanel">
          <div className="diagnosticsTitle">
            <strong>Prediction samples</strong>
            <span>predicted vs true</span>
          </div>
          <div className="predictionGrid">
            {samples.slice(0, 16).map((sample, index) => (
              <div className={`predictionSample ${sample.correct ? "correct" : "wrong"}`} key={`${sample.truth}-${sample.predicted}-${index}`}>
                <DigitCanvas pixels={sample.pixels} />
                <span>
                  {sample.predicted} / {sample.truth}
                </span>
                <small>{formatMaybePercent(sample.confidence)}</small>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function DigitCanvas({ pixels }: { pixels: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) {
      return;
    }
    const image = context.createImageData(28, 28);
    for (let index = 0; index < 28 * 28; index += 1) {
      const value = Math.max(0, Math.min(255, pixels[index] ?? 0));
      const offset = index * 4;
      image.data[offset] = value;
      image.data[offset + 1] = value;
      image.data[offset + 2] = value;
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [pixels]);
  return <canvas className="digitCanvas" width={28} height={28} ref={canvasRef} />;
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

function buildVariantPreview(project: TopologyProject, selectedNodeId: string, settings: VariantBuilderSettings): VariantPreview {
  const selectedNode = project.nodes.find((node) => node.id === selectedNodeId);
  const selectedIsResidual = selectedNode?.kind === "multi_branch_residual";
  const targetLabel = settings.scope === "all" ? "all residual blocks" : selectedIsResidual ? selectedNode.name : "select a residual block";
  const projectVariant: TopologyProject = {
    ...project,
    name: `${project.name} / Builder variant`,
    nodes: project.nodes.map((node) => {
      const shouldMutateResidual = node.kind === "multi_branch_residual" && (settings.scope === "all" || node.id === selectedNodeId);
      if (shouldMutateResidual) {
        const width = typeof node.parameters.out_channels === "number" ? Math.max(4, nearestMultiple(node.parameters.out_channels + settings.widthDelta, 4)) : node.parameters.out_channels;
        const dropPath =
          typeof node.parameters.drop_path === "number" ? Number(Math.max(0, Math.min(0.25, node.parameters.drop_path + settings.dropPathDelta)).toFixed(3)) : node.parameters.drop_path;
        return {
          ...node,
          parameters: {
            ...node.parameters,
            out_channels: width,
            drop_path: dropPath,
            use_se: settings.useSe,
          },
        };
      }
      if (node.kind === "pooling_fusion") {
        return { ...node, parameters: { ...node.parameters, mode: settings.poolingMode } };
      }
      return node;
    }),
    edges: project.edges.map((edge) => ({ ...edge })),
  };
  const diff = buildTopologyDiff(project, projectVariant);
  const labelParts = [
    settings.scope === "all" ? "All blocks" : selectedIsResidual ? selectedNode.name : "Variant",
    settings.widthDelta ? `width ${formatSignedInteger(settings.widthDelta)}` : "",
    settings.dropPathDelta ? `drop ${formatSignedDecimal(settings.dropPathDelta)}` : "",
    `SE ${settings.useSe ? "on" : "off"}`,
    settings.poolingMode === "gap" ? "GAP" : "GAP+GMP",
  ].filter(Boolean);
  return {
    project: projectVariant,
    label: labelParts.join(" / "),
    note: diff.summaryText,
    diff,
    targetLabel,
  };
}

function buildTopologyDiff(base: TopologyProject, project: TopologyProject): TopologyDiff {
  const baseResolution = resolveTopology(base);
  const resolution = resolveTopology(project);
  const baseNodes = new Map(base.nodes.map((node) => [node.id, node]));
  const projectNodes = new Map(project.nodes.map((node) => [node.id, node]));
  const lines: string[] = [];
  let changedFieldCount = 0;

  for (const node of project.nodes) {
    const baseNode = baseNodes.get(node.id);
    if (!baseNode) {
      lines.push(`Added ${kindLabel(node.kind)} ${node.name}`);
      continue;
    }
    if (baseNode.kind !== node.kind) {
      lines.push(`${node.name} kind ${kindLabel(baseNode.kind)} -> ${kindLabel(node.kind)}`);
      changedFieldCount += 1;
    }
    const keys = Array.from(new Set([...Object.keys(baseNode.parameters), ...Object.keys(node.parameters)])).sort();
    for (const key of keys) {
      const before = baseNode.parameters[key];
      const after = node.parameters[key];
      if (before !== after) {
        changedFieldCount += 1;
        lines.push(`${node.name} ${parameterLabel(key)} ${formatParamValue(before)} -> ${formatParamValue(after)}`);
      }
    }
  }

  for (const node of base.nodes) {
    if (!projectNodes.has(node.id)) {
      lines.push(`Removed ${kindLabel(node.kind)} ${node.name}`);
    }
  }

  const baseEdgeSet = new Set(base.edges.map(edgeKey));
  const projectEdgeSet = new Set(project.edges.map(edgeKey));
  const addedEdges = project.edges.filter((edge) => !baseEdgeSet.has(edgeKey(edge))).map(edgeLabel);
  const removedEdges = base.edges.filter((edge) => !projectEdgeSet.has(edgeKey(edge))).map(edgeLabel);
  for (const edge of addedEdges) {
    lines.push(`Added edge ${edge}`);
  }
  for (const edge of removedEdges) {
    lines.push(`Removed edge ${edge}`);
  }

  const addedNodes = project.nodes.filter((node) => !baseNodes.has(node.id)).map((node) => node.name);
  const removedNodes = base.nodes.filter((node) => !projectNodes.has(node.id)).map((node) => node.name);
  const totalChanges = changedFieldCount + addedNodes.length + removedNodes.length + addedEdges.length + removedEdges.length;
  return {
    lines,
    summaryText: summarizeDiffLines(lines),
    paramsDelta: resolution.totalParameters - baseResolution.totalParameters,
    flopsDelta: resolution.totalFlops - baseResolution.totalFlops,
    changedFieldCount,
    addedNodes,
    removedNodes,
    addedEdges,
    removedEdges,
    totalChanges,
  };
}

function buildWinnerSummary(queue: AblationQueueItem[]): WinnerSummary | null {
  const best = bestQueueVariant(queue);
  if (!best?.metrics) {
    return null;
  }
  const baseline = queue.find((item) => item.id === "current" && item.metrics) ?? queue.find((item) => item.metrics);
  const bestAccuracy = best.metrics.best_accuracy ?? best.metrics.test_accuracy;
  const baselineAccuracy = baseline?.metrics?.best_accuracy ?? baseline?.metrics?.test_accuracy;
  const diff = buildTopologyDiff(enhancedFiveBlockTopology, best.project);
  const accuracyText =
    typeof bestAccuracy === "number" && typeof baselineAccuracy === "number"
      ? `improved by ${formatMaybeSignedPercent(bestAccuracy - baselineAccuracy)} vs ${baseline?.label ?? "baseline"}`
      : `reached ${formatMaybePercent(bestAccuracy)}`;
  return {
    bestId: best.id,
    title: `${best.label} is best at ${formatMaybePercent(bestAccuracy)}`,
    detail: `${accuracyText}; params ${formatDeltaCompact(diff.paramsDelta)}, FLOPs ${formatDeltaCompact(diff.flopsDelta)}.`,
    diffSummary: diff.summaryText,
  };
}

function settingsFromMetrics(metrics: NonNullable<GeneratedTrainMetrics>, fallback: TrainSettings): TrainSettings {
  return {
    preset: "custom",
    epochs: metrics.epochs ?? fallback.epochs,
    trainLimit: metrics.train_limit ?? fallback.trainLimit,
    testLimit: metrics.test_limit ?? fallback.testLimit,
    batchSize: metrics.batch_size ?? fallback.batchSize,
    learningRate: metrics.learning_rate ?? fallback.learningRate,
    seed: metrics.seed ?? fallback.seed,
    cpu: fallback.cpu,
  };
}

function compareReplay(originalRunId: string, replayRunId: string, original: NonNullable<GeneratedTrainMetrics>, replay: GeneratedTrainMetrics): NonNullable<ReplayComparison> {
  const originalAccuracy = original.best_accuracy ?? original.test_accuracy ?? null;
  const replayAccuracy = replay?.best_accuracy ?? replay?.test_accuracy ?? null;
  const originalLoss = original.test_loss ?? null;
  const replayLoss = replay?.test_loss ?? null;
  const accuracyDelta = typeof originalAccuracy === "number" && typeof replayAccuracy === "number" ? replayAccuracy - originalAccuracy : null;
  const lossDelta = typeof originalLoss === "number" && typeof replayLoss === "number" ? replayLoss - originalLoss : null;
  const reproducible =
    typeof accuracyDelta === "number" &&
    typeof lossDelta === "number" &&
    Math.abs(accuracyDelta) <= REPLAY_TOLERANCE.accuracy &&
    Math.abs(lossDelta) <= REPLAY_TOLERANCE.loss;
  return {
    createdAt: new Date().toISOString(),
    originalRunId,
    replayRunId,
    originalAccuracy,
    replayAccuracy,
    originalLoss,
    replayLoss,
    accuracyDelta,
    lossDelta,
    reproducible,
    tolerance: REPLAY_TOLERANCE,
  };
}

function buildRunBundle({
  project,
  metrics,
  settings,
  resolution,
  diff,
  generatedFiles,
  queue,
  replayComparisons,
}: {
  project: TopologyProject;
  metrics: GeneratedTrainMetrics;
  settings: TrainSettings;
  resolution: ReturnType<typeof resolveTopology>;
  diff: TopologyDiff;
  generatedFiles: GeneratedFile[];
  queue: AblationQueueItem[];
  replayComparisons: NonNullable<ReplayComparison>[];
}) {
  return {
    exportedAt: new Date().toISOString(),
    topology: project,
    generatedFiles,
    trainSettings: metrics ? settingsFromMetrics(metrics, settings) : settings,
    metrics,
    diagnostics: metrics?.diagnostics ?? null,
    replayComparisons,
    manifest: {
      topologyId: topologyVersionId(project),
      params: resolution.totalParameters,
      flops: resolution.totalFlops,
      diff,
      winnerSummary: buildWinnerSummary(queue),
    },
  };
}

function replayComparisonsFromRuns(runHistory: GeneratedRunSummary[], latest: ReplayComparison) {
  const comparisons = [
    ...(latest ? [latest] : []),
    ...runHistory.map((run) => run.metrics.replay_comparison).filter((comparison): comparison is NonNullable<ReplayComparison> => Boolean(comparison)),
  ];
  const seen = new Set<string>();
  return comparisons.filter((comparison) => {
    if (seen.has(comparison.replayRunId)) {
      return false;
    }
    seen.add(comparison.replayRunId);
    return true;
  });
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
  const winnerSummary = buildWinnerSummary(queue);
  return queue.map((item) => {
    const resolution = resolveTopology(item.project);
    const baselineResolution = resolveTopology(enhancedFiveBlockTopology);
    const topologyDiff = buildTopologyDiff(enhancedFiveBlockTopology, item.project);
    const weakestClass = weakestPerClass(item.metrics?.diagnostics);
    const topConfusion = strongestConfusion(item.metrics?.diagnostics);
    return {
      label: item.label,
      note: item.note,
      topologyId: item.topologyId,
      status: item.status,
      runId: item.runId ?? "",
      runPath: item.runId ? `runs/${item.runId}` : "",
      diagnosticsPath: item.metrics?.diagnostics_path ?? (item.runId ? `runs/${item.runId}/diagnostics.json` : ""),
      topologyDiffSummary: topologyDiff.summaryText,
      changedFieldCount: topologyDiff.changedFieldCount,
      addedNodes: topologyDiff.addedNodes.join("; "),
      removedNodes: topologyDiff.removedNodes.join("; "),
      addedEdges: topologyDiff.addedEdges.join("; "),
      removedEdges: topologyDiff.removedEdges.join("; "),
      paramsDelta: resolution.totalParameters - baselineResolution.totalParameters,
      flopsDelta: resolution.totalFlops - baselineResolution.totalFlops,
      winnerSummary: winnerSummary?.bestId === item.id ? `${winnerSummary.title}; ${winnerSummary.detail}; ${winnerSummary.diffSummary}` : "",
      seed: item.metrics?.seed ?? null,
      learningRate: item.metrics?.learning_rate ?? null,
      accuracy: item.metrics?.test_accuracy ?? null,
      baselineAccuracy: item.metrics?.baseline_accuracy ?? null,
      bestAccuracy: item.metrics?.best_accuracy ?? item.metrics?.test_accuracy ?? null,
      bestEpoch: item.metrics?.best_epoch ?? null,
      accuracyDelta: item.metrics?.accuracy_delta ?? null,
      trainAccuracy: item.metrics?.train_accuracy ?? null,
      trainLoss: item.metrics?.train_loss ?? null,
      testLoss: item.metrics?.test_loss ?? null,
      finalBatchLoss: item.metrics?.final_batch_loss ?? null,
      weakestDigit: weakestClass?.digit ?? null,
      weakestDigitAccuracy: weakestClass?.accuracy ?? null,
      topConfusion: topConfusion ? `${topConfusion.truth}->${topConfusion.predicted}:${topConfusion.count}` : "",
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

function weakestPerClass(diagnostics?: RunDiagnostics | null) {
  return (diagnostics?.per_class_accuracy ?? [])
    .filter((item) => typeof item.accuracy === "number" && item.total > 0)
    .sort((left, right) => (left.accuracy ?? 1) - (right.accuracy ?? 1))[0];
}

function strongestConfusion(diagnostics?: RunDiagnostics | null): { truth: number; predicted: number; count: number } | null {
  let strongest: { truth: number; predicted: number; count: number } | null = null;
  const matrix = diagnostics?.confusion_matrix ?? [];
  for (let truth = 0; truth < matrix.length; truth += 1) {
    const row = matrix[truth] ?? [];
    for (let predicted = 0; predicted < row.length; predicted += 1) {
      const count = row[predicted] ?? 0;
      if (truth !== predicted && count > (strongest?.count ?? 0)) {
        strongest = { truth, predicted, count };
      }
    }
  }
  return strongest;
}

function edgeKey(edge: TopologyEdge) {
  return `${edge.source}->${edge.target}:${edge.branch ?? ""}`;
}

function edgeLabel(edge: TopologyEdge) {
  return `${edge.source} -> ${edge.target}${edge.branch ? ` (${edge.branch})` : ""}`;
}

function kindLabel(kind: NodeKind) {
  return kind.replaceAll("_", " ");
}

function parameterLabel(key: string) {
  const labels: Record<string, string> = {
    out_channels: "width",
    drop_path: "drop-path",
    use_se: "SE",
    se_reduction: "SE reduction",
    branch_count: "branches",
    mode: "pooling",
    hidden_features: "hidden",
    embedding_features: "embedding",
  };
  return labels[key] ?? key.replaceAll("_", " ");
}

function formatParamValue(value: boolean | number | string | undefined) {
  if (value === undefined) {
    return "missing";
  }
  if (typeof value === "boolean") {
    return value ? "on" : "off";
  }
  return String(value);
}

function summarizeDiffLines(lines: string[]) {
  if (lines.length === 0) {
    return "No architecture changes";
  }
  const visible = lines.slice(0, 2).join("; ");
  return lines.length > 2 ? `${visible}; +${lines.length - 2} more` : visible;
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
    "diagnosticsPath",
    "topologyDiffSummary",
    "changedFieldCount",
    "addedNodes",
    "removedNodes",
    "addedEdges",
    "removedEdges",
    "paramsDelta",
    "flopsDelta",
    "winnerSummary",
    "seed",
    "learningRate",
    "accuracy",
    "baselineAccuracy",
    "bestAccuracy",
    "bestEpoch",
    "accuracyDelta",
    "trainAccuracy",
    "trainLoss",
    "testLoss",
    "finalBatchLoss",
    "weakestDigit",
    "weakestDigitAccuracy",
    "topConfusion",
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

function getImprovementCheck(metrics: GeneratedTrainMetrics) {
  if (!metrics) {
    return null;
  }
  if (
    typeof metrics.baseline_accuracy !== "number" &&
    typeof metrics.first_batch_loss !== "number" &&
    typeof metrics.test_accuracy !== "number"
  ) {
    return null;
  }
  const baselineAccuracy = metrics.baseline_accuracy;
  const bestAccuracy = metrics.best_accuracy ?? metrics.test_accuracy;
  const accuracyDelta =
    typeof metrics.accuracy_delta === "number"
      ? metrics.accuracy_delta
      : typeof baselineAccuracy === "number" && typeof bestAccuracy === "number"
        ? bestAccuracy - baselineAccuracy
        : null;
  const trainLossDrop =
    typeof metrics.first_batch_loss === "number" && metrics.first_batch_loss > 0 && typeof metrics.final_batch_loss === "number"
      ? (metrics.first_batch_loss - metrics.final_batch_loss) / metrics.first_batch_loss
      : null;
  const validationLossTrend = getValidationLossTrend(metrics);
  const accuracyMoved = typeof accuracyDelta === "number" && accuracyDelta >= 0.01;
  const trainLossMoved = typeof trainLossDrop === "number" && trainLossDrop >= 0.05;
  const validationIsBetter = typeof validationLossTrend === "number" ? validationLossTrend <= -0.02 : false;
  const validationIsWorse = typeof validationLossTrend === "number" ? validationLossTrend >= 0.02 : false;
  const tone = accuracyMoved && trainLossMoved && !validationIsWorse ? "good" : accuracyMoved || trainLossMoved ? "warn" : "bad";
  const title =
    tone === "good"
      ? "Improvement check passed"
      : tone === "warn"
        ? "Training improved, validate carefully"
        : "No clear improvement yet";
  const validationLabel = validationIsBetter ? "val loss down" : validationIsWorse ? "val loss up" : "val loss flat";
  return {
    tone,
    title,
    detail: `Baseline ${formatMaybePercent(baselineAccuracy)} -> best ${formatMaybePercent(bestAccuracy)}; train loss ${formatLossDrop(trainLossDrop)}; ${validationLabel}.`,
  };
}

function getValidationLossTrend(metrics: NonNullable<GeneratedTrainMetrics>) {
  const history = metrics.epoch_history ?? [];
  if (history.length >= 2) {
    const first = history[0]?.test_loss;
    const last = history[history.length - 1]?.test_loss;
    return typeof first === "number" && typeof last === "number" ? last - first : null;
  }
  if (typeof metrics.baseline_test_loss === "number" && typeof metrics.test_loss === "number") {
    return metrics.test_loss - metrics.baseline_test_loss;
  }
  return null;
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

function formatLossDrop(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}% drop` : "-";
}

function formatDeltaCompact(value: number) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${formatCompactNumber(Math.abs(value))}`;
}

function formatSignedInteger(value: number) {
  return value > 0 ? `+${value}` : String(value);
}

function formatSignedDecimal(value: number) {
  if (value === 0) {
    return "0";
  }
  return `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatMaybeNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

function formatLearningRate(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value.toString() : "-";
}

function formatRunLabel(runId: string) {
  const [timestamp] = runId.split("_");
  if (!timestamp || timestamp.length < 15) {
    return runId;
  }
  return `${timestamp.slice(4, 6)}/${timestamp.slice(6, 8)} ${timestamp.slice(9, 11)}:${timestamp.slice(11, 13)}`;
}

function topologySummary(project: TopologyProject, resolution: ReturnType<typeof resolveTopology>) {
  const pooling = project.nodes.find((node) => node.kind === "pooling_fusion");
  const poolingLabel = pooling?.parameters.mode === "gap" ? "GAP" : "GAP + GMP";
  const auxLabel = resolution.auxiliaryHeads === 1 ? "1 aux head" : `${resolution.auxiliaryHeads} aux heads`;
  return `${resolution.residualPaths} residual blocks / ${auxLabel} / ${poolingLabel} / ${resolution.embeddingDimension ?? "-"}D embedding`;
}

function formatShortDate(value: string) {
  if (value === "builtin") {
    return "built-in";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return `${String(date.getMonth() + 1).padStart(2, "0")}/${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function architectureRunStats(architecture: ArchitectureSummary | undefined, runHistory: GeneratedRunSummary[]) {
  if (!architecture) {
    return { bestAccuracy: null, bestRunId: null, lastTrained: null };
  }
  const runs = runHistory.filter((run) => run.topologyId === architecture.topologyId || run.metrics.topology_id === architecture.topologyId);
  const best = runs
    .filter((run) => typeof (run.metrics.best_accuracy ?? run.metrics.test_accuracy) === "number")
    .sort((left, right) => (right.metrics.best_accuracy ?? right.metrics.test_accuracy ?? -1) - (left.metrics.best_accuracy ?? left.metrics.test_accuracy ?? -1))[0];
  const latest = runs
    .map((run) => run.updatedAt ?? run.metrics.updated_at ?? run.createdAt ?? run.metrics.created_at ?? "")
    .filter(Boolean)
    .sort((left, right) => right.localeCompare(left))[0];
  return {
    bestAccuracy: best?.metrics.best_accuracy ?? best?.metrics.test_accuracy ?? null,
    bestRunId: best?.runId ?? null,
    lastTrained: latest ?? null,
  };
}

function latestReplayStatusForTopology(topologyId: string, runHistory: GeneratedRunSummary[]) {
  const replayRun = runHistory
    .filter((run) => (run.topologyId === topologyId || run.metrics.topology_id === topologyId) && run.metrics.replay_comparison)
    .sort((left, right) => (right.updatedAt ?? right.metrics.updated_at ?? "").localeCompare(left.updatedAt ?? left.metrics.updated_at ?? ""))[0];
  if (!replayRun?.metrics.replay_comparison) {
    return "none";
  }
  return replayRun.metrics.replay_comparison.reproducible ? "reproducible" : "drifted";
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
