import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET() {
  const metricsPath = path.join(process.cwd(), "generated", "train_metrics.json");
  try {
    const metrics = JSON.parse(await readFile(metricsPath, "utf8"));
    return NextResponse.json({ metrics });
  } catch {
    return NextResponse.json({ metrics: null });
  }
}
