import { execFile } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

import { NextResponse } from "next/server";

import { generateFiles } from "@/lib/codegen";
import { parseTopologyProject } from "@/lib/topology";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const execFileAsync = promisify(execFile);

type TrainRequest = {
  project?: unknown;
  epochs?: unknown;
  trainLimit?: unknown;
  testLimit?: unknown;
  batchSize?: unknown;
  cpu?: unknown;
};

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as TrainRequest;
  const parsed = parseTopologyProject(payload.project);
  if (!parsed.project) {
    return NextResponse.json({ error: parsed.errors.join(" ") }, { status: 400 });
  }

  const generatedDir = path.join(process.cwd(), "generated");
  await mkdir(generatedDir, { recursive: true });
  for (const file of generateFiles(parsed.project)) {
    await writeFile(path.join(generatedDir, file.path), file.content.endsWith("\n") ? file.content : `${file.content}\n`, "utf8");
  }

  const epochs = clampInt(payload.epochs, 1, 5, 1);
  const trainLimit = clampInt(payload.trainLimit, 128, 10_000, 1024);
  const testLimit = clampInt(payload.testLimit, 128, 5_000, 512);
  const batchSize = clampInt(payload.batchSize, 16, 512, 128);
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
  ];
  if (payload.cpu !== false) {
    args.push("--cpu");
  }

  try {
    await execFileAsync(process.env.PYTHON ?? "python", args, {
      cwd: process.cwd(),
      timeout: 5 * 60 * 1000,
      maxBuffer: 1024 * 1024 * 8,
    });
    const metrics = JSON.parse(await readFile(path.join(generatedDir, "train_metrics.json"), "utf8"));
    return NextResponse.json({ metrics });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Generated training failed.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function clampInt(value: unknown, min: number, max: number, fallback: number) {
  const numberValue = typeof value === "number" && Number.isFinite(value) ? Math.trunc(value) : fallback;
  return Math.max(min, Math.min(max, numberValue));
}
