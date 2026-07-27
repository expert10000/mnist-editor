import { formatCompactNumber, formatShape, resolveTopology, type TensorShape, type TopologyNode, type TopologyProject } from "./topology";

export type GeneratedFile = {
  path: string;
  language: "json" | "markdown" | "python" | "yaml";
  content: string;
};

export type ArchitectureStage = {
  id: string;
  name: string;
  operator: string;
  inputShape?: TensorShape;
  outputShape?: TensorShape;
  parameters: Record<string, boolean | number | string>;
};

export type ArchitectureIr = {
  network: {
    name: string;
    version: string;
    inputShape: TensorShape;
    mainPath: ArchitectureStage[];
    auxiliaryHeads: Array<ArchitectureStage & { source: string }>;
    totals: {
      parameters: number;
      flops: number;
      branchCount: number;
      residualPaths: number;
      auxiliaryHeads: number;
      embeddingDimension?: number;
    };
    validation: {
      errors: string[];
      warnings: string[];
    };
  };
};

export function buildArchitectureIr(project: TopologyProject): ArchitectureIr {
  const resolution = resolveTopology(project);
  const nodeById = new Map(project.nodes.map((node) => [node.id, node]));
  const traceByNode = new Map(resolution.trace.map((entry) => [entry.nodeId, entry]));
  const mainPath = resolution.mainPath
    .map((nodeId) => nodeById.get(nodeId))
    .filter((node): node is TopologyNode => Boolean(node))
    .map((node) => {
      const trace = traceByNode.get(node.id);
      return {
        id: node.id,
        name: node.name,
        operator: node.kind,
        inputShape: trace?.inputShape,
        outputShape: trace?.outputShape,
        parameters: node.parameters,
      };
    });
  const auxiliaryHeads = project.edges
    .map((edge) => {
      const target = nodeById.get(edge.target);
      const source = nodeById.get(edge.source);
      if (!target || !source || target.kind !== "auxiliary_classifier") {
        return null;
      }
      const trace = traceByNode.get(target.id);
      return {
        id: target.id,
        name: target.name,
        operator: target.kind,
        source: source.id,
        inputShape: trace?.inputShape,
        outputShape: trace?.outputShape,
        parameters: target.parameters,
      };
    })
    .filter((head): head is NonNullable<typeof head> => Boolean(head));

  return {
    network: {
      name: slug(project.name),
      version: project.version,
      inputShape: project.inputShape,
      mainPath,
      auxiliaryHeads,
      totals: {
        parameters: resolution.totalParameters,
        flops: resolution.totalFlops,
        branchCount: resolution.branchCount,
        residualPaths: resolution.residualPaths,
        auxiliaryHeads: resolution.auxiliaryHeads,
        embeddingDimension: resolution.embeddingDimension,
      },
      validation: {
        errors: resolution.errors,
        warnings: resolution.warnings,
      },
    },
  };
}

export function generateFiles(project: TopologyProject): GeneratedFile[] {
  const ir = buildArchitectureIr(project);
  return [
    {
      path: "architecture.json",
      language: "json",
      content: JSON.stringify(ir, null, 2),
    },
    {
      path: "blocks.py",
      language: "python",
      content: generateBlocksPy(),
    },
    {
      path: "model.py",
      language: "python",
      content: generateModelPy(ir),
    },
    {
      path: "smoke_test.py",
      language: "python",
      content: generateSmokeTestPy(ir),
    },
    {
      path: "architecture.md",
      language: "markdown",
      content: generateArchitectureMarkdown(project, ir),
    },
    {
      path: "experiment_manifest.yaml",
      language: "yaml",
      content: generateExperimentManifest(project),
    },
  ];
}

function generateBlocksPy() {
  return `import torch
from torch import nn


class ConvBnGelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * mask.floor()


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class MultiBranchResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, branch_count=4, use_se=False, se_reduction=8, drop_path=0.0):
        super().__init__()
        if out_channels % branch_count != 0:
            raise ValueError("out_channels must be divisible by branch_count")
        branch_channels = out_channels // branch_count
        self.branches = nn.ModuleList([
            ConvBnGelu(in_channels, branch_channels, 3, stride, 1),
            ConvBnGelu(in_channels, branch_channels, 5, stride, 2),
            ConvBnGelu(in_channels, branch_channels, 3, stride, 2, dilation=2),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.GELU(),
            ),
        ])
        self.fusion = nn.Sequential(
            ConvBnGelu(out_channels, out_channels, 1, 1, 0),
            ConvBnGelu(out_channels, out_channels, 3, 1, 1),
        )
        self.se = SqueezeExcite(out_channels, se_reduction) if use_se else nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, x):
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        merged = self.fusion(merged)
        merged = self.se(merged)
        merged = self.drop_path(merged)
        return self.activation(merged + self.shortcut(x))


class PoolingFusion(nn.Module):
    def __init__(self, mode="gap_gmp"):
        super().__init__()
        self.mode = mode
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg = self.avg(x).flatten(1)
        if self.mode == "gap":
            return avg
        return torch.cat([avg, self.max(x).flatten(1)], dim=1)


class FeatureHead(nn.Module):
    def __init__(self, in_features, hidden_features=192, embedding_features=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, embedding_features),
        )

    def forward(self, x):
        return self.net(x)


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, classes),
        )

    def forward(self, x):
        return self.net(x)
`;
}

function generateModelPy(ir: ArchitectureIr) {
  const mainStages = ir.network.mainPath.filter((stage) => stage.operator !== "input");
  const auxHeads = ir.network.auxiliaryHeads;
  const lines: string[] = [
    "import torch",
    "from torch import nn",
    "",
    "from blocks import AuxiliaryClassifier, ConvBnGelu, FeatureHead, MultiBranchResidual, PoolingFusion",
    "",
    "",
    "class GeneratedMNISTModel(nn.Module):",
    "    def __init__(self):",
    "        super().__init__()",
  ];

  for (const stage of mainStages) {
    if (stage.operator === "conv_bn_gelu") {
      lines.push(
        `        self.${attr(stage.id)} = ConvBnGelu(${imageChannels(stage.inputShape)}, ${numberParam(stage, "out_channels", imageChannels(stage.outputShape))}, ${numberParam(stage, "kernel_size", 3)}, ${numberParam(stage, "stride", 1)}, ${numberParam(stage, "padding", 1)})`,
      );
    }
    if (stage.operator === "multi_branch_residual") {
      lines.push(
        `        self.${attr(stage.id)} = MultiBranchResidual(${imageChannels(stage.inputShape)}, ${numberParam(stage, "out_channels", imageChannels(stage.outputShape))}, stride=${numberParam(stage, "stride", 1)}, branch_count=${numberParam(stage, "branch_count", 4)}, use_se=${pythonBool(Boolean(stage.parameters.use_se))}, se_reduction=${numberParam(stage, "se_reduction", 8)}, drop_path=${numberParam(stage, "drop_path", 0)})`,
      );
    }
    if (stage.operator === "pooling_fusion") {
      lines.push(`        self.${attr(stage.id)} = PoolingFusion(mode="${stage.parameters.mode === "gap" ? "gap" : "gap_gmp"}")`);
    }
    if (stage.operator === "feature_head") {
      lines.push(
        `        self.${attr(stage.id)} = FeatureHead(${vectorFeatures(stage.inputShape)}, hidden_features=${numberParam(stage, "hidden_features", 192)}, embedding_features=${numberParam(stage, "embedding_features", 128)}, dropout=${numberParam(stage, "dropout", 0.2)})`,
      );
    }
    if (stage.operator === "classifier") {
      lines.push(`        self.${attr(stage.id)} = nn.Linear(${vectorFeatures(stage.inputShape)}, ${numberParam(stage, "classes", 10)})`);
    }
  }

  for (const head of auxHeads) {
    lines.push(`        self.${attr(head.id)} = AuxiliaryClassifier(${imageChannels(head.inputShape)}, classes=${numberParam(head, "classes", 10)})`);
  }

  lines.push("", "    def forward(self, x, return_features=False, return_auxiliary=False):", "        auxiliary_logits = None", "        embedding = None");

  for (const stage of mainStages) {
    lines.push(`        x = self.${attr(stage.id)}(x)`);
    for (const head of auxHeads.filter((candidate) => candidate.source === stage.id)) {
      lines.push(`        auxiliary_logits = self.${attr(head.id)}(x)`);
    }
    if (stage.operator === "feature_head") {
      lines.push("        embedding = x");
    }
  }

  lines.push(
    "        logits = x",
    "        if return_features and return_auxiliary:",
    "            return embedding, logits, auxiliary_logits",
    "        if return_features:",
    "            return embedding, logits",
    "        if return_auxiliary:",
    "            return logits, auxiliary_logits",
    "        return logits",
    "",
  );

  return lines.join("\n");
}

function generateSmokeTestPy(ir: ArchitectureIr) {
  const embedding = ir.network.totals.embeddingDimension ?? 128;
  const hasAuxiliary = ir.network.auxiliaryHeads.length > 0;
  return `import torch

from model import GeneratedMNISTModel


def smoke_test() -> None:
    model = GeneratedMNISTModel()
    images = torch.randn(8, 1, 28, 28)
    embedding, logits, auxiliary_logits = model(images, return_features=True, return_auxiliary=True)

    assert embedding.shape == (8, ${embedding})
    assert logits.shape == (8, 10)
${hasAuxiliary ? "    assert auxiliary_logits.shape == (8, 10)" : "    assert auxiliary_logits is None"}
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embedding).all()

    loss = logits.mean()
    if auxiliary_logits is not None:
        loss = loss + 0.1 * auxiliary_logits.mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    smoke_test()
`;
}

function generateArchitectureMarkdown(project: TopologyProject, ir: ArchitectureIr) {
  const rows = ir.network.mainPath
    .map((stage) => `| ${stage.name} | ${stage.operator} | ${formatShape(stage.inputShape)} | ${formatShape(stage.outputShape)} |`)
    .join("\n");
  const auxRows = ir.network.auxiliaryHeads
    .map((stage) => `| ${stage.name} | ${stage.source} | ${formatShape(stage.inputShape)} | ${formatShape(stage.outputShape)} |`)
    .join("\n");
  return `# ${project.name}

Version: ${project.version}

## Summary

- Parameters: ${formatCompactNumber(ir.network.totals.parameters)}
- Forward FLOPs: ${formatCompactNumber(ir.network.totals.flops)}
- Residual paths: ${ir.network.totals.residualPaths}
- Branches: ${ir.network.totals.branchCount}
- Auxiliary heads: ${ir.network.totals.auxiliaryHeads}
- Embedding dimension: ${ir.network.totals.embeddingDimension ?? "-"}

## Main Path

| Node | Operator | Input | Output |
|---|---|---|---|
${rows}

## Auxiliary Heads

| Head | Source | Input | Output |
|---|---|---|---|
${auxRows || "| None | - | - | - |"}

## Validation

${ir.network.validation.errors.length === 0 ? "No validation errors." : ir.network.validation.errors.map((error) => `- ${error}`).join("\n")}
`;
}

function generateExperimentManifest(project: TopologyProject) {
  const block5 = project.nodes.find((node) => node.id === "block5");
  const baseWidth = typeof block5?.parameters.out_channels === "number" ? block5.parameters.out_channels : 160;
  const widths = [baseWidth - 16, baseWidth, baseWidth + 16, baseWidth + 32].filter((width) => width > 0);
  const runCount = widths.length * 3 * 3;
  return `experiment_group: fiveblock_pooling_width_search
base_network: ${slug(project.name)}
expected_runs: ${runCount}
sweep:
  block5.out_channels:
${widths.map((width) => `    - ${width}`).join("\n")}
  pooling.type:
    - gap
    - gap_gmp
    - gem
seeds:
  - 1
  - 2
  - 3
`;
}

function numberParam(stage: ArchitectureStage, key: string, fallback: number) {
  const value = stage.parameters[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function imageChannels(shape: TensorShape | undefined) {
  return shape && shape.length === 4 ? shape[1] : 1;
}

function vectorFeatures(shape: TensorShape | undefined) {
  return shape && shape.length === 2 ? shape[1] : 1;
}

function pythonBool(value: boolean) {
  return value ? "True" : "False";
}

function attr(value: string) {
  return value.replace(/[^a-zA-Z0-9_]/g, "_");
}

function slug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}
