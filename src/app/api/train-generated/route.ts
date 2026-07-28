import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { generateFiles } from "@/lib/codegen";
import { parseTopologyProject, type TopologyProject } from "@/lib/topology";
import { topologyVersionId } from "@/lib/topologyVersion";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type TrainRequest = {
  project?: unknown;
  epochs?: unknown;
  trainLimit?: unknown;
  testLimit?: unknown;
  batchSize?: unknown;
  learningRate?: unknown;
  seed?: unknown;
  cpu?: unknown;
};

type TrainJob = {
  runId: string;
  topologyId: string;
  status: "running" | "complete" | "failed" | "cancelled";
  runDir: string;
  logPath: string;
  startedAt: string;
  finishedAt?: string;
  logs: string[];
  metrics?: unknown;
  child?: ChildProcessWithoutNullStreams;
  error?: string;
};

const globalJobs = globalThis as typeof globalThis & {
  generatedTrainingJobs?: Map<string, TrainJob>;
};

const jobs = globalJobs.generatedTrainingJobs ?? new Map<string, TrainJob>();
globalJobs.generatedTrainingJobs = jobs;

export async function GET(request: Request) {
  const url = new URL(request.url);
  const runId = url.searchParams.get("runId") ?? "";
  if (!runId) {
    return NextResponse.json({ error: "runId is required." }, { status: 400 });
  }

  const job = jobs.get(runId);
  if (job) {
    job.metrics = (await metricsWithDiagnostics((await readJson(path.join(job.runDir, "metrics.json"))) ?? job.metrics, job.runDir)) ?? job.metrics;
    const persistedLog = await readText(job.logPath);
    return NextResponse.json({
      runId: job.runId,
      topologyId: job.topologyId,
      status: job.status,
      startedAt: job.startedAt,
      finishedAt: job.finishedAt,
      metrics: job.metrics ?? null,
      logs: persistedLog ? trimLogs(persistedLog.split(/\r?\n/).filter(Boolean)) : job.logs,
      error: job.error,
    });
  }

  const runDir = projectPath("runs", runId);
  const metrics = await metricsWithDiagnostics(await readJson(path.join(runDir, "metrics.json")), runDir);
  if (!metrics) {
    return NextResponse.json({ error: "Run not found." }, { status: 404 });
  }

  const logText = await readText(path.join(runDir, "train.log"));
  return NextResponse.json({
    runId,
    topologyId: getString(metrics, "topology_id"),
    status: getString(metrics, "status") || "complete",
    metrics,
    logs: logText.split(/\r?\n/).filter(Boolean).slice(-160),
  });
}

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as TrainRequest;
  const parsed = parseTopologyProject(payload.project);
  if (!parsed.project) {
    return NextResponse.json({ error: parsed.errors.join(" ") }, { status: 400 });
  }

  const createdAt = new Date();
  const topologyId = topologyVersionId(parsed.project);
  const runId = `${timestampSegment(createdAt)}_${topologyId}`;
  const generatedDir = projectPath("generated");
  const runDir = projectPath("runs", runId);
  const logPath = path.join(runDir, "train.log");

  await mkdir(generatedDir, { recursive: true });
  await mkdir(runDir, { recursive: true });
  const generatedFiles = generateFiles(parsed.project);
  await writeGeneratedFiles(generatedDir, generatedFiles);
  await writeGeneratedFiles(runDir, generatedFiles);
  await writeProjectJson(generatedDir, parsed.project);
  await writeProjectJson(runDir, parsed.project);

  const epochs = clampInt(payload.epochs, 1, 5, 1);
  const trainLimit = clampInt(payload.trainLimit, 128, 10_000, 1024);
  const testLimit = clampInt(payload.testLimit, 128, 5_000, 512);
  const batchSize = clampInt(payload.batchSize, 16, 512, 128);
  const learningRate = clampFloat(payload.learningRate, 0.00001, 0.1, 0.001);
  const seed = clampInt(payload.seed, 0, 999_999, 7);
  const initialMetrics = {
    status: "running",
    run_id: runId,
    topology_id: topologyId,
    created_at: createdAt.toISOString(),
    updated_at: createdAt.toISOString(),
    epochs,
    train_limit: trainLimit,
    test_limit: testLimit,
    batch_size: batchSize,
    seed,
    learning_rate: learningRate,
    current_epoch: 0,
    current_batch: 0,
    total_batches: 0,
    first_batch_loss: null,
    final_batch_loss: null,
    final_batch_accuracy: null,
    train_loss: null,
    train_accuracy: null,
    baseline_test_loss: null,
    baseline_accuracy: null,
    test_loss: null,
    test_accuracy: null,
    best_accuracy: null,
    best_epoch: 0,
    accuracy_delta: null,
    epoch_history: [],
    diagnostics_path: path.join("runs", runId, "diagnostics.json"),
    checkpoint: path.join("runs", runId, "checkpoint.pt"),
    duration_seconds: 0,
    passed_smoke_rule: null,
  };
  await writeFile(path.join(runDir, "metrics.json"), JSON.stringify(initialMetrics, null, 2), "utf8");
  await writeFile(path.join(generatedDir, "train_metrics.json"), JSON.stringify(initialMetrics, null, 2), "utf8");
  await writeFile(logPath, `run ${runId} queued\n`, "utf8");

  const args = [
    "scripts/train_generated.py",
    "--epochs",
    String(epochs),
    "--train-limit",
    String(trainLimit),
    "--test-limit",
    String(testLimit),
    "--batch-size",
    String(batchSize),
    "--lr",
    String(learningRate),
    "--seed",
    String(seed),
    "--generated-dir",
    runDir,
    "--run-dir",
    runDir,
    "--metrics-path",
    path.join(runDir, "metrics.json"),
    "--latest-metrics-path",
    path.join(generatedDir, "train_metrics.json"),
    "--checkpoint-path",
    path.join(runDir, "checkpoint.pt"),
    "--latest-checkpoint-path",
    path.join(generatedDir, "generated_mnist.pt"),
    "--diagnostics-path",
    path.join(runDir, "diagnostics.json"),
    "--latest-diagnostics-path",
    path.join(generatedDir, "diagnostics.json"),
    "--run-id",
    runId,
    "--topology-id",
    topologyId,
    "--created-at",
    createdAt.toISOString(),
  ];
  if (payload.cpu !== false) {
    args.push("--cpu");
  }

  const job: TrainJob = {
    runId,
    topologyId,
    status: "running",
    runDir,
    logPath,
    startedAt: createdAt.toISOString(),
    logs: [`run ${runId} queued`],
    metrics: initialMetrics,
  };
  jobs.set(runId, job);

  const child = spawn(process.env.PYTHON ?? "python", args, {
    cwd: projectPath(),
    windowsHide: true,
  });
  job.child = child;

  child.stdout.on("data", (chunk: Buffer) => {
    appendLog(job, chunk.toString());
  });
  child.stderr.on("data", (chunk: Buffer) => {
    appendLog(job, chunk.toString());
  });
  child.on("error", (error) => {
    void finishJob(job, "failed", error.message);
  });
  child.on("close", (code) => {
    void finishJob(job, code === 0 ? "complete" : "failed", code === 0 ? undefined : `Training exited with code ${code}.`);
  });

  return NextResponse.json({ runId, topologyId, status: "running", metrics: initialMetrics, logs: job.logs }, { status: 202 });
}

export async function DELETE(request: Request) {
  const url = new URL(request.url);
  const runId = url.searchParams.get("runId") ?? "";
  if (!runId) {
    return NextResponse.json({ error: "runId is required." }, { status: 400 });
  }

  const job = jobs.get(runId);
  if (!job) {
    return NextResponse.json({ error: "Run not found or no longer active." }, { status: 404 });
  }
  if (job.status !== "running") {
    return NextResponse.json({ runId, status: job.status, metrics: job.metrics ?? null, logs: job.logs });
  }

  job.status = "cancelled";
  job.finishedAt = new Date().toISOString();
  job.error = "Cancelled by user.";
  appendLog(job, job.error);
  job.child?.kill();

  const metrics = ((await readJson(path.join(job.runDir, "metrics.json"))) as Record<string, unknown> | null) ?? {};
  const cancelledMetrics = { ...metrics, status: "cancelled", error: job.error, updated_at: job.finishedAt };
  await writeFile(path.join(job.runDir, "metrics.json"), JSON.stringify(cancelledMetrics, null, 2), "utf8");
  await writeFile(projectPath("generated", "train_metrics.json"), JSON.stringify(cancelledMetrics, null, 2), "utf8");
  job.metrics = cancelledMetrics;

  return NextResponse.json({ runId, status: "cancelled", metrics: cancelledMetrics, logs: job.logs, error: job.error });
}

async function finishJob(job: TrainJob, fallbackStatus: TrainJob["status"], error?: string) {
  if (job.status === "cancelled") {
    return;
  }
  const metrics = (await readJson(path.join(job.runDir, "metrics.json"))) as Record<string, unknown> | null;
  job.metrics = (await metricsWithDiagnostics(metrics, job.runDir)) ?? job.metrics;
  job.status =
    metrics?.status === "complete" || metrics?.status === "failed" || metrics?.status === "cancelled" ? metrics.status : fallbackStatus;
  job.finishedAt = new Date().toISOString();
  job.error = error;
  if (error) {
    appendLog(job, error);
  }

  if (job.status === "failed" && metrics && metrics.status !== "failed") {
    const failedMetrics = { ...metrics, status: "failed", error, updated_at: job.finishedAt };
    await writeFile(path.join(job.runDir, "metrics.json"), JSON.stringify(failedMetrics, null, 2), "utf8");
    await writeFile(projectPath("generated", "train_metrics.json"), JSON.stringify(failedMetrics, null, 2), "utf8");
    job.metrics = failedMetrics;
  }
}

function appendLog(job: TrainJob, value: string) {
  const lines = value.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    return;
  }
  job.logs = trimLogs([...job.logs, ...lines]);
  void appendFile(job.logPath, lines.map((line) => `${line}\n`).join(""), "utf8");
}

function trimLogs(logs: string[]) {
  return logs.slice(-160);
}

async function writeGeneratedFiles(outputDir: string, files: ReturnType<typeof generateFiles>) {
  await mkdir(outputDir, { recursive: true });
  for (const file of files) {
    await writeFile(path.join(outputDir, file.path), file.content.endsWith("\n") ? file.content : `${file.content}\n`, "utf8");
  }
}

async function writeProjectJson(outputDir: string, project: TopologyProject) {
  await writeFile(path.join(outputDir, "project.json"), JSON.stringify(project, null, 2), "utf8");
}

async function readJson(filePath: string) {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as unknown;
  } catch {
    return null;
  }
}

async function readText(filePath: string) {
  try {
    return await readFile(filePath, "utf8");
  } catch {
    return "";
  }
}

async function metricsWithDiagnostics(metrics: unknown, runDir: string) {
  if (!isRecord(metrics)) {
    return null;
  }
  const diagnostics = await readJson(path.join(runDir, "diagnostics.json"));
  return isRecord(diagnostics) ? { ...metrics, diagnostics } : metrics;
}

function getString(value: unknown, key: string) {
  return typeof value === "object" && value !== null && key in value && typeof (value as Record<string, unknown>)[key] === "string"
    ? ((value as Record<string, string>)[key] as string)
    : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function timestampSegment(date: Date) {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}

function clampInt(value: unknown, min: number, max: number, fallback: number) {
  const numberValue = typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : fallback;
  return Math.max(min, Math.min(max, numberValue));
}

function clampFloat(value: unknown, min: number, max: number, fallback: number) {
  const numberValue = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.max(min, Math.min(max, numberValue));
}
