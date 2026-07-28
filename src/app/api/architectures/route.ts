import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { predefinedArchitectures } from "@/lib/predefinedArchitectures";
import { enhancedFiveBlockTopology, parseTopologyProject, resolveTopology, type TopologyProject } from "@/lib/topology";
import { topologyVersionId } from "@/lib/topologyVersion";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type ArchitecturePayload = {
  id?: unknown;
  name?: unknown;
  notes?: unknown;
  tags?: unknown;
  parentId?: unknown;
  project?: unknown;
  sourceId?: unknown;
  archived?: unknown;
};

type ArchitectureRecord = {
  id: string;
  name: string;
  notes: string;
  tags: string[];
  parentId: string;
  topologyId: string;
  params: number;
  flops: number;
  createdAt: string;
  updatedAt: string;
  archived: boolean;
  locked: boolean;
  project: TopologyProject;
};

const BUILTIN_BASELINE_ID = "baseline-fiveblock";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const id = url.searchParams.get("id");
  const includeArchived = url.searchParams.get("archived") === "1";
  const architectures = await listArchitectures(includeArchived);
  if (id) {
    const architecture = architectures.find((item) => item.id === id);
    if (!architecture) {
      return NextResponse.json({ error: "Architecture not found." }, { status: 404 });
    }
    return NextResponse.json({ architecture });
  }
  return NextResponse.json({ architectures });
}

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as ArchitecturePayload;
  const parsed = parseTopologyProject(payload.project);
  if (!parsed.project) {
    return NextResponse.json({ error: parsed.errors.join(" ") || "Architecture project is invalid." }, { status: 400 });
  }

  const now = new Date().toISOString();
  const name = typeof payload.name === "string" && payload.name.trim() ? payload.name.trim() : parsed.project.name;
  const requestedId = typeof payload.id === "string" && payload.id.trim() ? slug(payload.id) : slug(name);
  const id = uniqueArchitectureId(requestedId);
  const record = buildArchitectureRecord({
    id,
    name,
    notes: typeof payload.notes === "string" ? payload.notes : "",
    tags: Array.isArray(payload.tags) ? payload.tags.filter((tag): tag is string => typeof tag === "string") : [],
    parentId: typeof payload.parentId === "string" ? payload.parentId : typeof payload.sourceId === "string" ? payload.sourceId : "",
    createdAt: now,
    updatedAt: now,
    archived: false,
    locked: false,
    project: { ...parsed.project, name },
  });

  await writeArchitecture(record);
  return NextResponse.json({ architecture: record });
}

export async function PATCH(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as ArchitecturePayload;
  const id = typeof payload.id === "string" ? payload.id : "";
  if (!safeId(id) || id === BUILTIN_BASELINE_ID) {
    return NextResponse.json({ error: "A saved architecture id is required." }, { status: 400 });
  }
  const existing = await readArchitecture(id);
  if (!existing) {
    return NextResponse.json({ error: "Architecture not found." }, { status: 404 });
  }
  if (existing.locked) {
    return NextResponse.json({ error: "Built-in architectures cannot be edited." }, { status: 400 });
  }

  const project = payload.project ? parseTopologyProject(payload.project).project : existing.project;
  if (!project) {
    return NextResponse.json({ error: "Architecture project is invalid." }, { status: 400 });
  }
  const next = buildArchitectureRecord({
    ...existing,
    name: typeof payload.name === "string" && payload.name.trim() ? payload.name.trim() : existing.name,
    notes: typeof payload.notes === "string" ? payload.notes : existing.notes,
    tags: Array.isArray(payload.tags) ? payload.tags.filter((tag): tag is string => typeof tag === "string") : existing.tags,
    parentId: typeof payload.parentId === "string" ? payload.parentId : existing.parentId,
    archived: typeof payload.archived === "boolean" ? payload.archived : existing.archived,
    updatedAt: new Date().toISOString(),
    project,
  });
  await writeArchitecture(next);
  return NextResponse.json({ architecture: next });
}

export async function DELETE(request: Request) {
  const url = new URL(request.url);
  const id = url.searchParams.get("id") ?? "";
  if (!safeId(id) || id === BUILTIN_BASELINE_ID) {
    return NextResponse.json({ error: "A saved architecture id is required." }, { status: 400 });
  }
  const existing = await readArchitecture(id);
  if (!existing) {
    return NextResponse.json({ error: "Architecture not found." }, { status: 404 });
  }
  if (existing.locked) {
    return NextResponse.json({ error: "Built-in architectures cannot be archived." }, { status: 400 });
  }
  const archived = { ...existing, archived: true, updatedAt: new Date().toISOString() };
  await writeArchitecture(archived);
  return NextResponse.json({ architecture: archived });
}

async function listArchitectures(includeArchived: boolean) {
  const root = projectPath("architectures");
  const names = await readdir(root).catch(() => []);
  const saved = (
    await Promise.all(
      names.map(async (name) => {
        const directory = path.join(root, name);
        const directoryStat = await stat(directory).catch(() => null);
        if (!directoryStat?.isDirectory()) {
          return null;
        }
        return readArchitecture(name);
      }),
    )
  )
    .filter((architecture): architecture is ArchitectureRecord => Boolean(architecture))
    .filter((architecture) => includeArchived || !architecture.archived);

  const savedSorted = saved.sort((left, right) => Number(left.archived) - Number(right.archived) || right.updatedAt.localeCompare(left.updatedAt));
  return [baselineRecord(), ...predefinedRecords(), ...savedSorted];
}

function baselineRecord() {
  return buildArchitectureRecord({
    id: BUILTIN_BASELINE_ID,
    name: enhancedFiveBlockTopology.name,
    notes: "Built-in baseline architecture.",
    tags: ["baseline"],
    parentId: "",
    createdAt: "builtin",
    updatedAt: "builtin",
    archived: false,
    locked: true,
    project: enhancedFiveBlockTopology,
  });
}

function predefinedRecords() {
  return predefinedArchitectures.map((architecture) =>
    buildArchitectureRecord({
      id: architecture.id,
      name: architecture.name,
      notes: architecture.notes,
      tags: architecture.tags,
      parentId: architecture.parentId,
      createdAt: "builtin",
      updatedAt: "builtin",
      archived: false,
      locked: true,
      project: architecture.project,
    }),
  );
}

async function readArchitecture(id: string) {
  if (!safeId(id)) {
    return null;
  }
  const payload = await readJson(path.join(projectPath("architectures", id), "architecture.json"));
  return isRecord(payload) ? normalizeArchitecture(payload) : null;
}

async function writeArchitecture(record: ArchitectureRecord) {
  const directory = projectPath("architectures", record.id);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "architecture.json"), JSON.stringify(record, null, 2), "utf8");
}

function buildArchitectureRecord(input: Omit<ArchitectureRecord, "topologyId" | "params" | "flops">) {
  const resolution = resolveTopology(input.project);
  return {
    ...input,
    topologyId: topologyVersionId(input.project),
    params: resolution.totalParameters,
    flops: resolution.totalFlops,
  };
}

function normalizeArchitecture(value: Record<string, unknown>): ArchitectureRecord | null {
  const parsed = parseTopologyProject(value.project);
  if (!parsed.project || typeof value.id !== "string" || typeof value.name !== "string") {
    return null;
  }
  return buildArchitectureRecord({
    id: safeId(value.id) ? value.id : slug(value.name),
    name: value.name,
    notes: typeof value.notes === "string" ? value.notes : "",
    tags: Array.isArray(value.tags) ? value.tags.filter((tag): tag is string => typeof tag === "string") : [],
    parentId: typeof value.parentId === "string" ? value.parentId : "",
    createdAt: typeof value.createdAt === "string" ? value.createdAt : new Date().toISOString(),
    updatedAt: typeof value.updatedAt === "string" ? value.updatedAt : new Date().toISOString(),
    archived: value.archived === true,
    locked: value.locked === true,
    project: parsed.project,
  });
}

async function readJson(filePath: string) {
  try {
    return JSON.parse(await readFile(filePath, "utf8")) as unknown;
  } catch {
    return null;
  }
}

function uniqueArchitectureId(base: string) {
  return `${base}-${Date.now().toString(36)}`;
}

function safeId(value: string) {
  return /^[a-z0-9][a-z0-9-]*$/.test(value);
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "architecture";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function projectPath(...segments: string[]) {
  return path.join(/* turbopackIgnore: true */ process.cwd(), ...segments);
}
