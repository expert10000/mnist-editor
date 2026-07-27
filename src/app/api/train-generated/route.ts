import { spawn } from "node:child_process";
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { generateFiles } from "@/lib/codegen";
import { parseTopologyProject } from "@/lib/topology";
import { topologyVersionId } from "@/lib/topologyVersion";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type TrainRequest = {
  project?: unknown;
  epochs?: unknown;
  trainLimit?: unknown;
  testLimit?: unknown;
  batchSize?: unknown;
  cpu?: unknown;
};

type TrainJob = {
  runId: string;
  topologyId: string;
  status: "running" | "complete" | "failed";
  runDir: string;
  logPath: string;
  startedAt: string;
  finishedAt?: string;
  logs: string[];
  metrics?: unknown;
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
    job.metrics = (await readJson(path.join(job.runDir, "metrics.json"))) ?? job.metrics;
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
  const metrics = await readJson(path.join(runDir, "metrics.json"));
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

  const epochs = clampInt(payload.epochs, 1, 5, 1);
  const trainLimit = clampInt(payload.trainLimit, 128, 10_000, 1024);
  const testLimit = clampInt(payload.testLimit, 128, 5_000, 512);
  const batchSize = clampInt(payload.batchSize, 16, 512, 128);
  const initialMetrics = {
    status: "running",
    run_id: runId,
    topology_id: topologyId,
    created_at: createdAt.toISOString(),
    updated_at: createdAt.toISOString(),
    epochs,
    train_limit: trainLimit,
    test_limit: testLimit,
    seed: 7,
    current_epoch: 0,
    current_batch: 0,
    total_batches: 0,
    first_batch_loss: null,
    final_batch_loss: null,
    train_loss: null,
    test_loss: null,
    test_accuracy: null,
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

async function finishJob(job: TrainJob, fallbackStatus: TrainJob["status"], error?: string) {
  const metrics = (await readJson(path.join(job.runDir, "metrics.json"))) as Record<string, unknown> | null;
  job.metrics = metrics ?? job.metrics;
  job.status = metrics?.status === "complete" || metrics?.status === "failed" ? metrics.status : fallbackStatus;
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

function getString(value: unknown, key: string) {
  return typeof value === "object" && value !== null && key in value && typeof (value as Record<string, unknown>)[key] === "string"
    ? ((value as Record<string, string>)[key] as string)
    : undefined;
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
