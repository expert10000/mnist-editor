import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

import { generateFiles } from "../src/lib/codegen";
import { enhancedFiveBlockTopology } from "../src/lib/topology";

const outputDir = path.resolve(process.cwd(), "generated");
const files = generateFiles(enhancedFiveBlockTopology);

async function main() {
  await mkdir(outputDir, { recursive: true });

  for (const file of files) {
    const outputPath = path.join(outputDir, file.path);
    await writeFile(outputPath, file.content.endsWith("\n") ? file.content : `${file.content}\n`, "utf8");
    console.log(`wrote ${path.relative(process.cwd(), outputPath)}`);
  }
}

void main();
