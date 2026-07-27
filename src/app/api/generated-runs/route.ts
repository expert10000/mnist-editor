import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  const runsDir = projectPath("runs");
  const names = await readdir(runsDir).catch(() => []);
  const runs = (
    await Promise.all(
      names.map(async (name) => {
        const runDir = path.join(runsDir, name);
        const runStat = await stat(runDir).catch(() => null);
        if (!runStat?.isDirectory()) {
          return null;
        }
        const metrics = await readJson(path.join(runDir, "metrics.json"));
        if (!metrics) {
          return null;
        }
        return {
          runId: name,
          topologyId: stringField(metrics, "topology_id"),
          createdAt: stringField(metrics, "created_at") ?? runStat.birthtime.toISOString(),
          updatedAt: stringField(metrics, "updated_at") ?? runStat.mtime.toISOString(),
          status: stringField(metrics, "status") ?? "complete",
          metrics,
        };
      }),
    )
  )
    .filter((run): run is NonNullable<typeof run> => Boolean(run))
    .sort((left, right) => (right.createdAt ?? "").localeCompare(left.createdAt ?? ""));

  return NextResponse.json({ runs });
}

async function readJson(filePath: string) {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function stringField(value: Record<string, unknown>, key: string) {
  return typeof value[key] === "string" ? value[key] : undefined;
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}
