import type { TopologyProject } from "./topology";

type CanonicalNode = {
  id: string;
  kind: string;
  parameters: Record<string, boolean | number | string>;
};

type CanonicalEdge = {
  source: string;
  target: string;
  branch: string;
};

export function topologyVersionId(project: TopologyProject) {
  const canonical = {
    inputShape: project.inputShape,
    nodes: project.nodes
      .map<CanonicalNode>((node) => ({
        id: node.id,
        kind: node.kind,
        parameters: sortRecord(node.parameters),
      }))
      .sort((left, right) => left.id.localeCompare(right.id)),
    edges: project.edges
      .map<CanonicalEdge>((edge) => ({
        source: edge.source,
        target: edge.target,
        branch: edge.branch ?? "",
      }))
      .sort((left, right) => `${left.source}:${left.target}:${left.branch}`.localeCompare(`${right.source}:${right.target}:${right.branch}`)),
  };
  return fnv1a(JSON.stringify(canonical)).toString(16).padStart(8, "0");
}

function sortRecord(record: Record<string, boolean | number | string>) {
  return Object.fromEntries(Object.entries(record).sort(([left], [right]) => left.localeCompare(right)));
}

function fnv1a(value: string) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}
