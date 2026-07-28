import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { generateFiles } from "@/lib/codegen";
import { parseTopologyProject } from "@/lib/topology";
import { topologyVersionId } from "@/lib/topologyVersion";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type WinnerPayload = {
  experimentName?: unknown;
  variantId?: unknown;
  label?: unknown;
  project?: unknown;
  metrics?: unknown;
  diagnostics?: unknown;
  report?: unknown;
  manifest?: unknown;
};

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as WinnerPayload;
  const parsed = parseTopologyProject(payload.project);
  if (!parsed.project) {
    return NextResponse.json({ error: parsed.errors.join(" ") || "Winner project is invalid." }, { status: 400 });
  }

  const experimentName = typeof payload.experimentName === "string" && payload.experimentName.trim() ? payload.experimentName.trim() : "experiment";
  const directory = projectPath("experiments", slug(experimentName), "winner");
  const generatedFiles = generateFiles(parsed.project);
  const topologyId = topologyVersionId(parsed.project);
  const now = new Date().toISOString();
  const manifest = {
    lockedAt: now,
    experimentName,
    variantId: typeof payload.variantId === "string" ? payload.variantId : "",
    label: typeof payload.label === "string" ? payload.label : "",
    topologyId,
    checkpoint: stringField(payload.metrics, "checkpoint") ?? "",
    ...(isRecord(payload.manifest) ? payload.manifest : {}),
  };

  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "project.json"), JSON.stringify(parsed.project, null, 2), "utf8");
  await writeFile(path.join(directory, "metrics.json"), JSON.stringify(isRecord(payload.metrics) ? payload.metrics : {}, null, 2), "utf8");
  await writeFile(path.join(directory, "diagnostics.json"), JSON.stringify(isRecord(payload.diagnostics) ? payload.diagnostics : {}, null, 2), "utf8");
  await writeFile(path.join(directory, "manifest.json"), JSON.stringify(manifest, null, 2), "utf8");
  await writeFile(path.join(directory, "report.json"), JSON.stringify({ exportedAt: now, report: payload.report ?? null }, null, 2), "utf8");
  for (const file of generatedFiles) {
    await writeFile(path.join(directory, file.path), file.content.endsWith("\n") ? file.content : `${file.content}\n`, "utf8");
  }

  return NextResponse.json({ winner: { directory: path.join("experiments", slug(experimentName), "winner"), manifest } });
}

function stringField(value: unknown, key: string) {
  return isRecord(value) && typeof value[key] === "string" ? value[key] : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "experiment";
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}
