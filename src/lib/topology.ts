export type TensorShape = ["B", number] | ["B", number, number, number];

export type NodeKind =
  | "input"
  | "conv_bn_gelu"
  | "multi_branch_residual"
  | "auxiliary_classifier"
  | "pooling_fusion"
  | "feature_head"
  | "classifier";

export type TopologyNode = {
  id: string;
  name: string;
  kind: NodeKind;
  description: string;
  position: { x: number; y: number };
  parameters: Record<string, boolean | number | string>;
};

export type TopologyEdge = {
  id: string;
  source: string;
  target: string;
  branch?: string;
};

export type TopologyProject = {
  name: string;
  version: string;
  inputShape: TensorShape;
  nodes: TopologyNode[];
  edges: TopologyEdge[];
};

export type TraceEntry = {
  nodeId: string;
  name: string;
  kind: NodeKind;
  inputShape?: TensorShape;
  outputShape?: TensorShape;
  parameters: number;
  flops: number;
  errors: string[];
  warnings: string[];
};

export type TopologyResolution = {
  trace: TraceEntry[];
  errors: string[];
  warnings: string[];
  totalParameters: number;
  totalFlops: number;
  branchCount: number;
  residualPaths: number;
  auxiliaryHeads: number;
  embeddingDimension?: number;
};

export const enhancedFiveBlockTopology: TopologyProject = {
  name: "Enhanced Five-Block MNIST V1",
  version: "0.1.0",
  inputShape: ["B", 1, 28, 28],
  nodes: [
    {
      id: "input",
      name: "MNIST image",
      kind: "input",
      description: "Single-channel 28 x 28 digit image.",
      position: { x: 60, y: 82 },
      parameters: { channels: 1, height: 28, width: 28 },
    },
    {
      id: "stem",
      name: "Stem",
      kind: "conv_bn_gelu",
      description: "3 x 3 convolution with batch norm and GELU.",
      position: { x: 260, y: 82 },
      parameters: { out_channels: 32, kernel_size: 3, stride: 1, padding: 1 },
    },
    {
      id: "block1",
      name: "Block 1",
      kind: "multi_branch_residual",
      description: "Four-branch residual block.",
      position: { x: 460, y: 82 },
      parameters: { out_channels: 48, stride: 1, branch_count: 4, use_se: false, se_reduction: 8, drop_path: 0 },
    },
    {
      id: "block2",
      name: "Block 2",
      kind: "multi_branch_residual",
      description: "Downsampling four-branch residual block.",
      position: { x: 660, y: 82 },
      parameters: { out_channels: 80, stride: 2, branch_count: 4, use_se: false, se_reduction: 8, drop_path: 0.01 },
    },
    {
      id: "block3",
      name: "Block 3",
      kind: "multi_branch_residual",
      description: "SE-enabled block with an auxiliary classifier output.",
      position: { x: 860, y: 82 },
      parameters: { out_channels: 96, stride: 1, branch_count: 4, use_se: true, se_reduction: 8, drop_path: 0.02 },
    },
    {
      id: "auxiliary_head",
      name: "Auxiliary head",
      kind: "auxiliary_classifier",
      description: "Auxiliary classifier attached to Block 3.",
      position: { x: 860, y: 244 },
      parameters: { classes: 10 },
    },
    {
      id: "block4",
      name: "Block 4",
      kind: "multi_branch_residual",
      description: "Second downsampling SE residual block.",
      position: { x: 1060, y: 82 },
      parameters: { out_channels: 144, stride: 2, branch_count: 4, use_se: true, se_reduction: 8, drop_path: 0.03 },
    },
    {
      id: "block5",
      name: "Block 5",
      kind: "multi_branch_residual",
      description: "Final SE residual block before pooling.",
      position: { x: 1260, y: 82 },
      parameters: { out_channels: 160, stride: 1, branch_count: 4, use_se: true, se_reduction: 8, drop_path: 0.05 },
    },
    {
      id: "pooling",
      name: "GAP + GMP",
      kind: "pooling_fusion",
      description: "Global average and max pooling concatenated into one vector.",
      position: { x: 1460, y: 82 },
      parameters: { mode: "gap_gmp" },
    },
    {
      id: "feature_head",
      name: "Feature head",
      kind: "feature_head",
      description: "Projection to a reusable ensemble embedding.",
      position: { x: 1660, y: 82 },
      parameters: { hidden_features: 192, embedding_features: 128, dropout: 0.2 },
    },
    {
      id: "classifier",
      name: "Classifier",
      kind: "classifier",
      description: "Final ten-class digit classifier.",
      position: { x: 1860, y: 82 },
      parameters: { classes: 10 },
    },
  ],
  edges: [
    { id: "input_stem", source: "input", target: "stem" },
    { id: "stem_block1", source: "stem", target: "block1" },
    { id: "block1_block2", source: "block1", target: "block2" },
    { id: "block2_block3", source: "block2", target: "block3" },
    { id: "block3_aux", source: "block3", target: "auxiliary_head", branch: "auxiliary" },
    { id: "block3_block4", source: "block3", target: "block4" },
    { id: "block4_block5", source: "block4", target: "block5" },
    { id: "block5_pooling", source: "block5", target: "pooling" },
    { id: "pooling_feature_head", source: "pooling", target: "feature_head" },
    { id: "feature_head_classifier", source: "feature_head", target: "classifier" },
  ],
};

const mainPath = ["input", "stem", "block1", "block2", "block3", "block4", "block5", "pooling", "feature_head", "classifier"];

export function resolveTopology(project: TopologyProject): TopologyResolution {
  const nodeById = new Map(project.nodes.map((node) => [node.id, node]));
  const trace: TraceEntry[] = [];
  const shapeByNode = new Map<string, TensorShape>();
  let currentShape: TensorShape | undefined = project.inputShape;

  for (const nodeId of mainPath) {
    const node = nodeById.get(nodeId);
    if (!node) {
      continue;
    }
    const entry = resolveNode(node, currentShape);
    trace.push(entry);
    if (entry.outputShape) {
      shapeByNode.set(node.id, entry.outputShape);
      currentShape = entry.outputShape;
    }
  }

  const auxiliaryHead = nodeById.get("auxiliary_head");
  const block3Shape = shapeByNode.get("block3");
  if (auxiliaryHead) {
    trace.splice(5, 0, resolveNode(auxiliaryHead, block3Shape));
  }

  const errors = trace.flatMap((entry) => entry.errors.map((error) => `${entry.name}: ${error}`));
  const warnings = trace.flatMap((entry) => entry.warnings.map((warning) => `${entry.name}: ${warning}`));
  const featureHead = trace.find((entry) => entry.nodeId === "feature_head");
  const residuals = project.nodes.filter((node) => node.kind === "multi_branch_residual");

  return {
    trace,
    errors,
    warnings,
    totalParameters: trace.reduce((sum, entry) => sum + entry.parameters, 0),
    totalFlops: trace.reduce((sum, entry) => sum + entry.flops, 0),
    branchCount: residuals.reduce((sum, node) => sum + asNumber(node.parameters.branch_count, 0), 0),
    residualPaths: residuals.length,
    auxiliaryHeads: project.nodes.filter((node) => node.kind === "auxiliary_classifier").length,
    embeddingDimension: featureHead?.outputShape?.[1],
  };
}

function resolveNode(node: TopologyNode, inputShape?: TensorShape): TraceEntry {
  const errors: string[] = [];
  const warnings: string[] = [];
  let outputShape: TensorShape | undefined = inputShape;
  let parameters = 0;
  let flops = 0;

  if (!inputShape) {
    return traceEntry(node, inputShape, outputShape, parameters, flops, ["missing input tensor"], warnings);
  }

  if (node.kind === "input") {
    outputShape = inputShape;
  }

  if (node.kind === "conv_bn_gelu") {
    const [batch, inChannels, height, width] = imageShape(inputShape, errors);
    const outChannels = asNumber(node.parameters.out_channels, inChannels);
    const kernel = asNumber(node.parameters.kernel_size, 3);
    const stride = asNumber(node.parameters.stride, 1);
    const padding = asNumber(node.parameters.padding, Math.floor(kernel / 2));
    const outHeight = convDim(height, kernel, stride, padding);
    const outWidth = convDim(width, kernel, stride, padding);
    outputShape = [batch, outChannels, outHeight, outWidth];
    parameters = convParams(inChannels, outChannels, kernel) + batchNormParams(outChannels);
    flops = convFlops(inChannels, outChannels, outHeight, outWidth, kernel);
  }

  if (node.kind === "multi_branch_residual") {
    const [batch, inChannels, height, width] = imageShape(inputShape, errors);
    const outChannels = asNumber(node.parameters.out_channels, inChannels);
    const branchCount = asNumber(node.parameters.branch_count, 4);
    const stride = asNumber(node.parameters.stride, 1);
    const useSe = Boolean(node.parameters.use_se);
    const reduction = asNumber(node.parameters.se_reduction, 8);
    const outHeight = convDim(height, 3, stride, 1);
    const outWidth = convDim(width, 3, stride, 1);
    const branchChannels = outChannels / branchCount;
    const validBranchChannels = Math.max(1, Math.floor(branchChannels));

    outputShape = [batch, outChannels, outHeight, outWidth];

    if (!Number.isInteger(branchChannels)) {
      errors.push(`${outChannels} output channels cannot be split into ${branchCount} equal branches`);
    }
    if (branchCount < 1) {
      errors.push("branch count must be positive");
    }
    if (outHeight < 1 || outWidth < 1) {
      errors.push("downsampling collapses the spatial dimensions");
    }

    const branchParams =
      convParams(inChannels, validBranchChannels, 3) +
      convParams(inChannels, validBranchChannels, 5) +
      convParams(inChannels, validBranchChannels, 3) +
      inChannels * 3 * 3 +
      convParams(inChannels, validBranchChannels, 1);
    const fusionParams = convParams(outChannels, outChannels, 1) + convParams(outChannels, outChannels, 3);
    const shortcutParams = stride !== 1 || inChannels !== outChannels ? convParams(inChannels, outChannels, 1) : 0;
    const seParams = useSe ? outChannels * Math.max(1, Math.floor(outChannels / reduction)) * 2 : 0;

    parameters = branchParams + fusionParams + shortcutParams + seParams + batchNormParams(outChannels) * 2;
    flops =
      convFlops(inChannels, validBranchChannels, outHeight, outWidth, 3) +
      convFlops(inChannels, validBranchChannels, outHeight, outWidth, 5) +
      convFlops(inChannels, validBranchChannels, outHeight, outWidth, 3) +
      inChannels * 3 * 3 * outHeight * outWidth * 2 +
      convFlops(inChannels, validBranchChannels, outHeight, outWidth, 1) +
      convFlops(outChannels, outChannels, outHeight, outWidth, 1) +
      convFlops(outChannels, outChannels, outHeight, outWidth, 3);
  }

  if (node.kind === "auxiliary_classifier") {
    const [batch, channels] = imageShape(inputShape, errors);
    const classes = asNumber(node.parameters.classes, 10);
    outputShape = [batch, classes];
    parameters = linearParams(channels, classes);
    if (classes !== 10) {
      errors.push("auxiliary classifier must produce 10 logits for MNIST");
    }
  }

  if (node.kind === "pooling_fusion") {
    const [batch, channels] = imageShape(inputShape, errors);
    outputShape = [batch, channels * (node.parameters.mode === "gap" ? 1 : 2)];
  }

  if (node.kind === "feature_head") {
    const [batch, features] = vectorShape(inputShape, errors);
    const hidden = asNumber(node.parameters.hidden_features, 192);
    const embedding = asNumber(node.parameters.embedding_features, 128);
    outputShape = [batch, embedding];
    parameters = linearParams(features, hidden) + hidden * 2 + linearParams(hidden, embedding);
    if (embedding <= 0) {
      errors.push("embedding dimension must be positive");
    }
  }

  if (node.kind === "classifier") {
    const [batch, features] = vectorShape(inputShape, errors);
    const classes = asNumber(node.parameters.classes, 10);
    outputShape = [batch, classes];
    parameters = linearParams(features, classes);
    if (classes !== 10) {
      errors.push("final classifier must produce 10 logits for MNIST");
    }
  }

  return traceEntry(node, inputShape, outputShape, parameters, flops, errors, warnings);
}

function traceEntry(
  node: TopologyNode,
  inputShape: TensorShape | undefined,
  outputShape: TensorShape | undefined,
  parameters: number,
  flops: number,
  errors: string[],
  warnings: string[],
): TraceEntry {
  return {
    nodeId: node.id,
    name: node.name,
    kind: node.kind,
    inputShape,
    outputShape,
    parameters,
    flops,
    errors,
    warnings,
  };
}

function imageShape(shape: TensorShape, errors: string[]): ["B", number, number, number] {
  if (shape.length !== 4) {
    errors.push("expected image tensor [B, C, H, W]");
    return ["B", shape[1] ?? 1, 1, 1];
  }
  return shape;
}

function vectorShape(shape: TensorShape, errors: string[]): ["B", number] {
  if (shape.length !== 2) {
    errors.push("expected vector tensor [B, F]");
    return ["B", shape[1] ?? 1];
  }
  return shape;
}

function convDim(input: number, kernel: number, stride: number, padding: number, dilation = 1) {
  return Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1);
}

function convParams(inChannels: number, outChannels: number, kernel: number, bias = false) {
  return outChannels * inChannels * kernel * kernel + (bias ? outChannels : 0);
}

function linearParams(inFeatures: number, outFeatures: number, bias = true) {
  return inFeatures * outFeatures + (bias ? outFeatures : 0);
}

function batchNormParams(channels: number) {
  return channels * 2;
}

function convFlops(inChannels: number, outChannels: number, height: number, width: number, kernel: number) {
  return inChannels * outChannels * Math.max(1, height) * Math.max(1, width) * kernel * kernel * 2;
}

function asNumber(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function formatShape(shape?: TensorShape) {
  return shape ? `[${shape.join(", ")}]` : "-";
}

export function formatCompactNumber(value: number) {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`;
  }
  return value.toLocaleString();
}
