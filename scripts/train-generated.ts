import { spawn } from "node:child_process";
import { appendFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { generateFiles } from "../src/lib/codegen";
import { enhancedFiveBlockTopology } from "../src/lib/topology";
import { topologyVersionId } from "../src/lib/topologyVersion";

const root = process.cwd();
const generatedDir = path.join(root, "generated");
const topologyId = topologyVersionId(enhancedFiveBlockTopology);
const runId = `${new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z")}_${topologyId}`;
const runDir = path.join(root, "runs", runId);
const logPath = path.join(runDir, "train.log");

async function main() {
  await mkdir(generatedDir, { recursive: true });
  await mkdir(runDir, { recursive: true });
  const files = generateFiles(enhancedFiveBlockTopology);
  await writeGeneratedFiles(generatedDir, files);
  await writeGeneratedFiles(runDir, files);
  await writeFile(logPath, `run ${runId} queued\n`, "utf8");

  const args = [
    "scripts/train_generated.py",
    "--epochs",
    "1",
    "--train-limit",
    "1024",
    "--test-limit",
    "512",
    "--batch-size",
    "128",
    "--lr",
    "0.001",
    "--seed",
    "7",
    "--cpu",
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
  ];

  const child = spawn(process.env.PYTHON ?? "python", args, {
    cwd: root,
    windowsHide: true,
  });

  child.stdout.on("data", (chunk: Buffer) => {
    process.stdout.write(chunk);
    void appendFile(logPath, chunk, "utf8");
  });
  child.stderr.on("data", (chunk: Buffer) => {
    process.stderr.write(chunk);
    void appendFile(logPath, chunk, "utf8");
  });
  child.on("close", (code) => {
    process.exitCode = code ?? 1;
  });
}

async function writeGeneratedFiles(outputDir: string, files: ReturnType<typeof generateFiles>) {
  for (const file of files) {
    await writeFile(path.join(outputDir, file.path), file.content.endsWith("\n") ? file.content : `${file.content}\n`, "utf8");
  }
}

void main();
