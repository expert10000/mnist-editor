import { enhancedFiveBlockTopology, type TopologyEdge, type TopologyNode, type TopologyProject } from "./topology";

export type PredefinedArchitecture = {
  id: string;
  name: string;
  notes: string;
  tags: string[];
  parentId: string;
  project: TopologyProject;
};

type ProjectMutator = (project: TopologyProject) => TopologyProject;

export const predefinedArchitectures: PredefinedArchitecture[] = [
  preset("preset-compact-three-block", "Compact Three-Block MNIST", "Small fast architecture for quick checks and low-FLOP comparisons.", ["preset", "compact", "fast"], (project) =>
    pipe(
      withoutAuxiliary(relayout(replaceMainPath(project, ["input", "stem", "block1", "block2", "block3", "pooling", "feature_head", "classifier"]), 180)),
      withWidths([24, 32, 48], 24),
      withPooling("gap"),
      withFeatureHead(96, 64, 0.1),
      withSe([false, false, false]),
      withDropPath([0, 0, 0.01]),
    )
  ),
  preset("preset-compact-five-block", "Compact Five-Block MNIST", "Five residual blocks with narrow channels for cheaper architecture sweeps.", ["preset", "compact", "five-block"], (project) =>
    pipe(
      withoutAuxiliary(project),
      withWidths([24, 32, 48, 64, 80], 24),
      withPooling("gap"),
      withFeatureHead(112, 72, 0.12),
      withSe([false, false, false, true, true]),
      withDropPath([0, 0, 0.01, 0.015, 0.02]),
    ),
  ),
  preset("preset-no-se-control", "No-SE Control", "Baseline-sized control model with squeeze-excitation disabled everywhere.", ["preset", "control", "no-se"], (project) =>
    pipe(project, withSe([false, false, false, false, false]), withDropPath([0, 0.005, 0.01, 0.015, 0.02]), withFeatureHead(192, 128, 0.18)),
  ),
  preset("preset-gap-lite", "GAP Lite Head", "Uses average pooling only and a leaner embedding head to test pooling/head cost.", ["preset", "pooling", "lite"], (project) =>
    pipe(project, withPooling("gap"), withFeatureHead(128, 96, 0.15), withDropPath([0, 0.005, 0.015, 0.02, 0.03])),
  ),
  preset("preset-wide-se", "Wide SE MNIST", "Wider residual stages with SE enabled for stronger but slower comparisons.", ["preset", "wide", "se"], (project) =>
    pipe(project, withWidths([64, 96, 128, 192, 224], 40), withSe([true, true, true, true, true]), withFeatureHead(256, 160, 0.22), withDropPath([0, 0.01, 0.02, 0.035, 0.05])),
  ),
  preset("preset-regularized", "Regularized DropPath", "Baseline width with stronger dropout and drop-path for overfit checks.", ["preset", "regularized"], (project) =>
    pipe(project, withSe([false, true, true, true, true]), withFeatureHead(192, 128, 0.32), withDropPath([0.01, 0.025, 0.04, 0.06, 0.08])),
  ),
  preset("preset-deep-six-block", "Deep Six-Block MNIST", "Adds a sixth residual block after the baseline stack for depth experiments.", ["preset", "deep", "six-block"], (project) =>
    pipe(addSixthBlock(project), withWidths([48, 80, 96, 144, 160, 192], 32), withSe([false, false, true, true, true, true]), withDropPath([0, 0.01, 0.02, 0.035, 0.05, 0.06]), withFeatureHead(224, 144, 0.22)),
  ),
  preset("preset-large-embedding", "Large Embedding Head", "Baseline trunk with a larger feature projection for representation experiments.", ["preset", "embedding"], (project) =>
    pipe(project, withFeatureHead(320, 192, 0.25), withDropPath([0, 0.01, 0.02, 0.03, 0.05])),
  ),
];

function preset(id: string, name: string, notes: string, tags: string[], mutate: ProjectMutator): PredefinedArchitecture {
  return {
    id,
    name,
    notes,
    tags,
    parentId: "baseline-fiveblock",
    project: { ...mutate(cloneProject(enhancedFiveBlockTopology)), name },
  };
}

function pipe(project: TopologyProject, ...mutators: ProjectMutator[]) {
  return mutators.reduce((current, mutate) => mutate(current), project);
}

function withWidths(widths: number[], stemWidth: number): ProjectMutator {
  return (project) => ({
    ...project,
    nodes: project.nodes.map((node) => {
      if (node.id === "stem") {
        return updateParameters(node, { out_channels: stemWidth });
      }
      const blockNumber = blockIndex(node.id);
      if (blockNumber !== null && widths[blockNumber] !== undefined) {
        return updateParameters(node, { out_channels: widths[blockNumber] });
      }
      return node;
    }),
  });
}

function withSe(values: boolean[]): ProjectMutator {
  return (project) => ({
    ...project,
    nodes: project.nodes.map((node) => {
      const blockNumber = blockIndex(node.id);
      return blockNumber !== null && values[blockNumber] !== undefined ? updateParameters(node, { use_se: values[blockNumber] }) : node;
    }),
  });
}

function withDropPath(values: number[]): ProjectMutator {
  return (project) => ({
    ...project,
    nodes: project.nodes.map((node) => {
      const blockNumber = blockIndex(node.id);
      return blockNumber !== null && values[blockNumber] !== undefined ? updateParameters(node, { drop_path: values[blockNumber] }) : node;
    }),
  });
}

function withPooling(mode: "gap" | "gap_gmp"): ProjectMutator {
  return (project) => ({
    ...project,
    nodes: project.nodes.map((node) => (node.id === "pooling" ? { ...updateParameters(node, { mode }), name: mode === "gap" ? "GAP" : "GAP + GMP" } : node)),
  });
}

function withFeatureHead(hidden: number, embedding: number, dropout: number): ProjectMutator {
  return (project) => ({
    ...project,
    nodes: project.nodes.map((node) => (node.id === "feature_head" ? updateParameters(node, { hidden_features: hidden, embedding_features: embedding, dropout }) : node)),
  });
}

function withoutAuxiliary(project: TopologyProject): TopologyProject {
  const auxiliaryIds = new Set(project.nodes.filter((node) => node.kind === "auxiliary_classifier").map((node) => node.id));
  return {
    ...project,
    nodes: project.nodes.filter((node) => !auxiliaryIds.has(node.id)),
    edges: project.edges.filter((edge) => edge.branch !== "auxiliary" && !auxiliaryIds.has(edge.target)),
  };
}

function replaceMainPath(project: TopologyProject, path: string[]): TopologyProject {
  const keep = new Set(path);
  return {
    ...project,
    nodes: project.nodes.filter((node) => keep.has(node.id) || node.kind === "auxiliary_classifier"),
    edges: [
      ...path.slice(0, -1).map<TopologyEdge>((source, index) => ({ id: `${source}_${path[index + 1]}`, source, target: path[index + 1] })),
      ...project.edges.filter((edge) => edge.branch === "auxiliary"),
    ],
  };
}

function addSixthBlock(project: TopologyProject): TopologyProject {
  const block5 = project.nodes.find((node) => node.id === "block5");
  const block6: TopologyNode = {
    id: "block6",
    name: "Block 6",
    kind: "multi_branch_residual",
    description: "Extra SE residual block for deeper topology experiments.",
    position: { x: 1460, y: 82 },
    parameters: { ...(block5?.parameters ?? {}), out_channels: 192, stride: 1, drop_path: 0.06 },
  };
  return relayout(
    {
      ...project,
      nodes: [
        ...project.nodes.map((node) => {
          if (["pooling", "feature_head", "classifier"].includes(node.id)) {
            return { ...node, position: { ...node.position, x: node.position.x + 200 } };
          }
          return node;
        }),
        block6,
      ],
      edges: [
        ...project.edges.filter((edge) => edge.id !== "block5_pooling"),
        { id: "block5_block6", source: "block5", target: "block6" },
        { id: "block6_pooling", source: "block6", target: "pooling" },
      ],
    },
    200,
  );
}

function relayout(project: TopologyProject, spacing: number): TopologyProject {
  const mainIds = ["input", "stem", "block1", "block2", "block3", "block4", "block5", "block6", "pooling", "feature_head", "classifier"].filter((id) =>
    project.nodes.some((node) => node.id === id),
  );
  const xById = new Map(mainIds.map((id, index) => [id, 60 + index * spacing]));
  return {
    ...project,
    nodes: project.nodes.map((node) => {
      if (xById.has(node.id)) {
        return { ...node, position: { x: xById.get(node.id) ?? node.position.x, y: 82 } };
      }
      if (node.kind === "auxiliary_classifier") {
        return { ...node, position: { x: xById.get("block3") ?? node.position.x, y: 244 } };
      }
      return node;
    }),
  };
}

function updateParameters(node: TopologyNode, parameters: Record<string, boolean | number | string>): TopologyNode {
  return { ...node, parameters: { ...node.parameters, ...parameters } };
}

function blockIndex(id: string) {
  const match = /^block(\d+)$/.exec(id);
  return match ? Number(match[1]) - 1 : null;
}

function cloneProject(project: TopologyProject): TopologyProject {
  return {
    ...project,
    inputShape: [...project.inputShape] as TopologyProject["inputShape"],
    nodes: project.nodes.map((node) => ({ ...node, position: { ...node.position }, parameters: { ...node.parameters } })),
    edges: project.edges.map((edge) => ({ ...edge })),
  };
}
