import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type ExperimentPayload = {
  name?: unknown;
  preset?: unknown;
  queue?: unknown;
  reportRows?: unknown;
  bestVariantId?: unknown;
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const name = url.searchParams.get("name");
  if (name) {
    const session = await readJson(path.join(experimentDir(name), "session.json"));
    if (!session) {
      return NextResponse.json({ error: "Experiment not found." }, { status: 404 });
    }
    return NextResponse.json({ experiment: session });
  }

  const root = projectPath("experiments");
  const names = await readdir(root).catch(() => []);
  const experiments = (
    await Promise.all(
      names.map(async (entry) => {
        const directory = path.join(root, entry);
        const directoryStat = await stat(directory).catch(() => null);
        if (!directoryStat?.isDirectory()) {
          return null;
        }
        const session = await readJson(path.join(directory, "session.json"));
        if (!isRecord(session)) {
          return null;
        }
        return {
          name: stringField(session, "name") ?? entry,
          preset: stringField(session, "preset") ?? "",
          updatedAt: stringField(session, "updatedAt") ?? directoryStat.mtime.toISOString(),
          bestVariantId: stringField(session, "bestVariantId") ?? "",
          variantCount: Array.isArray(session.queue) ? session.queue.length : 0,
        };
      }),
    )
  )
    .filter((experiment): experiment is NonNullable<typeof experiment> => Boolean(experiment))
    .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

  return NextResponse.json({ experiments });
}

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as ExperimentPayload;
  const name = typeof payload.name === "string" && payload.name.trim() ? payload.name.trim() : "experiment";
  const safeName = slug(name);
  const now = new Date().toISOString();
  const directory = experimentDir(safeName);
  const reportRows = Array.isArray(payload.reportRows) ? payload.reportRows : [];
  const session = {
    name: safeName,
    displayName: name,
    preset: typeof payload.preset === "string" ? payload.preset : "",
    createdAt: now,
    updatedAt: now,
    bestVariantId: typeof payload.bestVariantId === "string" ? payload.bestVariantId : "",
    queue: Array.isArray(payload.queue) ? payload.queue : [],
    reportRows,
  };

  await mkdir(directory, { recursive: true });
  const existing = await readJson(path.join(directory, "session.json"));
  if (isRecord(existing) && typeof existing.createdAt === "string") {
    session.createdAt = existing.createdAt;
  }
  await writeFile(path.join(directory, "session.json"), JSON.stringify(session, null, 2), "utf8");
  await writeFile(path.join(directory, "report.json"), JSON.stringify({ exportedAt: now, rows: reportRows }, null, 2), "utf8");
  await writeFile(path.join(directory, "report.csv"), toCsv(reportRows), "utf8");

  return NextResponse.json({ experiment: session });
}

async function readJson(filePath: string) {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as unknown;
  } catch {
    return null;
  }
}

function toCsv(rows: unknown[]) {
  const headers = [
    "label",
    "note",
    "topologyId",
    "status",
    "runId",
    "runPath",
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
    "parameters",
    "flops",
    "projectName",
  ];
  return [headers.join(","), ...rows.map((row) => headers.map((header) => csvCell(isRecord(row) ? row[header] : "")).join(","))].join("\n");
}

function csvCell(value: unknown) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function stringField(value: Record<string, unknown>, key: string) {
  return typeof value[key] === "string" ? value[key] : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function experimentDir(name: string) {
  return projectPath("experiments", slug(name));
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "experiment";
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}
