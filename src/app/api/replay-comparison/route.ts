import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type ReplayComparisonPayload = {
  replayRunId?: unknown;
  comparison?: unknown;
};

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as ReplayComparisonPayload;
  const replayRunId = typeof payload.replayRunId === "string" ? payload.replayRunId : "";
  if (!safeRunId(replayRunId)) {
    return NextResponse.json({ error: "A valid replayRunId is required." }, { status: 400 });
  }
  if (!isRecord(payload.comparison)) {
    return NextResponse.json({ error: "A replay comparison object is required." }, { status: 400 });
  }

  const runDir = projectPath("runs", replayRunId);
  const comparison = { ...payload.comparison, savedAt: new Date().toISOString() };
  await mkdir(runDir, { recursive: true });
  await writeFile(path.join(runDir, "replay_comparison.json"), JSON.stringify(comparison, null, 2), "utf8");
  return NextResponse.json({ comparison });
}

function safeRunId(value: string) {
  return /^[a-zA-Z0-9_.-]+$/.test(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}
