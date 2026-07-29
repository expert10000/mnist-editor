import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type ComparisonPayload = {
  name?: unknown;
  selectedArchitectureIds?: unknown;
  trainSettings?: unknown;
  results?: unknown;
  reportRows?: unknown;
  winnerArchitectureId?: unknown;
  winnerSummary?: unknown;
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const name = url.searchParams.get("name");
  if (name) {
    const session = await readJson(path.join(comparisonDir(name), "session.json"));
    if (!session) {
      return NextResponse.json({ error: "Comparison not found." }, { status: 404 });
    }
    return NextResponse.json({ comparison: session });
  }

  const root = projectPath("comparisons");
  const names = await readdir(root).catch(() => []);
  const comparisons = (
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
          displayName: stringField(session, "displayName") ?? stringField(session, "name") ?? entry,
          updatedAt: stringField(session, "updatedAt") ?? directoryStat.mtime.toISOString(),
          winnerArchitectureId: stringField(session, "winnerArchitectureId") ?? "",
          architectureCount: Array.isArray(session.selectedArchitectureIds) ? session.selectedArchitectureIds.length : 0,
          bestAccuracy: bestAccuracyFromRows(Array.isArray(session.reportRows) ? session.reportRows : []),
        };
      }),
    )
  )
    .filter((comparison): comparison is NonNullable<typeof comparison> => Boolean(comparison))
    .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

  return NextResponse.json({ comparisons });
}

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as ComparisonPayload;
  const name = typeof payload.name === "string" && payload.name.trim() ? payload.name.trim() : "architecture-comparison";
  const safeName = slug(name);
  const now = new Date().toISOString();
  const directory = comparisonDir(safeName);
  const reportRows = Array.isArray(payload.reportRows) ? payload.reportRows : [];
  const session = {
    name: safeName,
    displayName: name,
    createdAt: now,
    updatedAt: now,
    selectedArchitectureIds: Array.isArray(payload.selectedArchitectureIds) ? payload.selectedArchitectureIds.filter((id): id is string => typeof id === "string") : [],
    trainSettings: isRecord(payload.trainSettings) ? payload.trainSettings : null,
    results: isRecord(payload.results) ? payload.results : {},
    winnerArchitectureId: typeof payload.winnerArchitectureId === "string" ? payload.winnerArchitectureId : "",
    winnerSummary: isRecord(payload.winnerSummary) ? payload.winnerSummary : null,
    reportRows,
  };

  await mkdir(directory, { recursive: true });
  const existing = await readJson(path.join(directory, "session.json"));
  if (isRecord(existing) && typeof existing.createdAt === "string") {
    session.createdAt = existing.createdAt;
  }
  await writeFile(path.join(directory, "session.json"), JSON.stringify(session, null, 2), "utf8");
  await writeFile(path.join(directory, "report.json"), JSON.stringify({ exportedAt: now, winnerSummary: session.winnerSummary, rows: reportRows }, null, 2), "utf8");
  await writeFile(path.join(directory, "report.csv"), toCsv(reportRows), "utf8");

  return NextResponse.json({ comparison: session });
}

async function readJson(filePath: string) {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as unknown;
  } catch {
    return null;
  }
}

function bestAccuracyFromRows(rows: unknown[]) {
  return rows.reduce<number | null>((best, row) => {
    const accuracy = isRecord(row) && typeof row.bestAccuracy === "number" ? row.bestAccuracy : null;
    if (accuracy === null) {
      return best;
    }
    return best === null ? accuracy : Math.max(best, accuracy);
  }, null);
}

function toCsv(rows: unknown[]) {
  const headers = [
    "architectureId",
    "architectureName",
    "topologyId",
    "status",
    "runId",
    "runPath",
    "params",
    "flops",
    "blocks",
    "auxHeads",
    "pooling",
    "embedding",
    "bestAccuracy",
    "finalAccuracy",
    "trainLoss",
    "testLoss",
    "learningRate",
    "seed",
    "epochs",
    "trainSamples",
    "testSamples",
    "replayStatus",
    "winner",
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

function comparisonDir(name: string) {
  return projectPath("comparisons", slug(name));
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "architecture-comparison";
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}
