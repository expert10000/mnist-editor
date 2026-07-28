import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const metricsPath = projectPath("generated", "train_metrics.json");
  try {
    const metrics = JSON.parse(await readFile(metricsPath, "utf8"));
    const diagnostics = await readDiagnostics();
    return NextResponse.json({ metrics: diagnostics ? { ...metrics, diagnostics } : metrics });
  } catch {
    return NextResponse.json({ metrics: null });
  }
}

async function readDiagnostics() {
  try {
    return JSON.parse(await readFile(projectPath("generated", "diagnostics.json"), "utf8"));
  } catch {
    return null;
  }
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}
