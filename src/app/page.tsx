"use client";

import { BrainCircuit, CheckCircle2, FileCode2, FlaskConical, RefreshCw, Table2, TriangleAlert, XCircle } from "lucide-react";
import { Fragment, useMemo, useState, type ReactNode } from "react";

import {
  enhancedFiveBlockTopology,
  formatCompactNumber,
  formatShape,
  resolveTopology,
  type TopologyProject,
} from "@/lib/topology";

type BottomPanel = "trace" | "validation" | "ir" | "experiments";

export default function Home() {
  const [project, setProject] = useState<TopologyProject>(enhancedFiveBlockTopology);
  const [selectedNodeId, setSelectedNodeId] = useState("block3");
  const [bottomPanel, setBottomPanel] = useState<BottomPanel>("trace");
  const resolution = useMemo(() => resolveTopology(project), [project]);
  const selectedNode = project.nodes.find((node) => node.id === selectedNodeId) ?? project.nodes[0];
  const selectedTrace = resolution.trace.find((entry) => entry.nodeId === selectedNode.id);
  const mainNodes = project.nodes.filter((node) => node.id !== "auxiliary_head");
  const auxiliaryNode = project.nodes.find((node) => node.id === "auxiliary_head");
  const irPreview = useMemo(() => buildIntermediatePreview(project), [project]);
  const experimentPreview = useMemo(() => buildExperimentPreview(project), [project]);

  function updateNodeParameter(nodeId: string, key: string, value: boolean | number | string) {
    setProject((current) => ({
      ...current,
      nodes: current.nodes.map((node) =>
        node.id === nodeId ? { ...node, parameters: { ...node.parameters, [key]: value } } : node,
      ),
    }));
  }

  function restoreTemplate() {
    setProject(enhancedFiveBlockTopology);
    setSelectedNodeId("block3");
    setBottomPanel("trace");
  }

  return (
    <main className="appShell">
      <header className="topbar">
        <div>
          <p className="eyebrow">MNIST topology compiler</p>
          <h1>{project.name}</h1>
        </div>
        <button className="iconButton" type="button" onClick={restoreTemplate} title="Restore template" aria-label="Restore template">
          <RefreshCw size={18} />
        </button>
      </header>

      <section className="summaryGrid">
        <SummaryItem label="Parameters" value={formatCompactNumber(resolution.totalParameters)} />
        <SummaryItem label="Forward FLOPs" value={formatCompactNumber(resolution.totalFlops)} />
        <SummaryItem label="Branches" value={resolution.branchCount.toLocaleString()} />
        <SummaryItem label="Embedding" value={`${resolution.embeddingDimension ?? "-"}D`} />
        <SummaryItem
          label="Validation"
          value={resolution.errors.length === 0 ? "Pass" : `${resolution.errors.length} issue${resolution.errors.length === 1 ? "" : "s"}`}
          tone={resolution.errors.length === 0 ? "good" : "bad"}
        />
      </section>

      <section className="workspace">
        <aside className="libraryPanel" aria-label="Node library">
          <div className="panelHeader">
            <strong>Templates</strong>
            <BrainCircuit size={18} />
          </div>
          <button className="templateButton active" type="button" onClick={restoreTemplate}>
            <strong>Enhanced Five-Block MNIST V1</strong>
            <span>5 residual blocks / auxiliary head / GAP + GMP / 128D embedding</span>
          </button>
          <div className="nodePalette">
            {["Input", "Conv", "Branch", "Residual", "SE", "Pool", "Head", "Output"].map((label) => (
              <span key={label}>{label}</span>
            ))}
          </div>
        </aside>

        <section className="canvasPanel" aria-label="Topology canvas">
          <div className="canvasViewport">
            <div className="topologyBoard">
              <svg className="topologyEdges" viewBox="0 0 2040 380" aria-hidden="true">
                {mainNodes.slice(0, -1).map((node, index) => {
                  const next = mainNodes[index + 1];
                  return (
                    <line
                      key={`${node.id}-${next.id}`}
                      x1={node.position.x + 158}
                      y1={node.position.y + 44}
                      x2={next.position.x}
                      y2={next.position.y + 44}
                    />
                  );
                })}
                {auxiliaryNode ? <polyline points="1018,126 1018,184 860,184 860,244" /> : null}
              </svg>
              {project.nodes.map((node) => {
                const trace = resolution.trace.find((entry) => entry.nodeId === node.id);
                const selected = selectedNodeId === node.id;
                const invalid = Boolean(trace?.errors.length);
                return (
                  <button
                    className={`topologyNode ${selected ? "selected" : ""} ${invalid ? "invalid" : ""}`}
                    key={node.id}
                    type="button"
                    onClick={() => setSelectedNodeId(node.id)}
                    style={{ left: node.position.x, top: node.position.y }}
                  >
                    <span>{node.kind.replaceAll("_", " ")}</span>
                    <strong>{node.name}</strong>
                    <small>{formatShape(trace?.outputShape)}</small>
                  </button>
                );
              })}
            </div>
          </div>
        </section>

        <aside className="inspectorPanel" aria-label="Selected node inspector">
          <div className="panelHeader">
            <strong>Inspector</strong>
            {selectedTrace?.errors.length ? <XCircle size={18} /> : <CheckCircle2 size={18} />}
          </div>
          <div className="inspectorTitle">
            <span>{selectedNode.kind.replaceAll("_", " ")}</span>
            <h2>{selectedNode.name}</h2>
            <p>{selectedNode.description}</p>
          </div>
          <div className="shapeGrid">
            <span>Input</span>
            <strong>{formatShape(selectedTrace?.inputShape)}</strong>
            <span>Output</span>
            <strong>{formatShape(selectedTrace?.outputShape)}</strong>
            <span>Params</span>
            <strong>{formatCompactNumber(selectedTrace?.parameters ?? 0)}</strong>
            <span>FLOPs</span>
            <strong>{formatCompactNumber(selectedTrace?.flops ?? 0)}</strong>
          </div>
          <div className="parameterEditor">
            {Object.entries(selectedNode.parameters).map(([key, value]) => (
              <label key={key}>
                <span>{key.replaceAll("_", " ")}</span>
                {typeof value === "boolean" ? (
                  <input
                    type="checkbox"
                    checked={value}
                    onChange={(event) => updateNodeParameter(selectedNode.id, key, event.target.checked)}
                  />
                ) : typeof value === "number" ? (
                  <input
                    type="number"
                    step={key.includes("drop") || key.includes("weight") ? 0.01 : 1}
                    value={value}
                    onChange={(event) => updateNodeParameter(selectedNode.id, key, Number(event.target.value))}
                  />
                ) : key === "mode" ? (
                  <select value={value} onChange={(event) => updateNodeParameter(selectedNode.id, key, event.target.value)}>
                    <option value="gap">GAP</option>
                    <option value="gap_gmp">GAP + GMP</option>
                  </select>
                ) : (
                  <input value={value} onChange={(event) => updateNodeParameter(selectedNode.id, key, event.target.value)} />
                )}
              </label>
            ))}
          </div>
        </aside>
      </section>

      <section className="bottomPanel">
        <div className="panelTabs" aria-label="Compiler panels">
          <PanelButton icon={<Table2 size={17} />} label="Shape Trace" value="trace" selected={bottomPanel} onSelect={setBottomPanel} />
          <PanelButton
            icon={<TriangleAlert size={17} />}
            label="Validation"
            value="validation"
            selected={bottomPanel}
            onSelect={setBottomPanel}
          />
          <PanelButton icon={<FileCode2 size={17} />} label="IR Preview" value="ir" selected={bottomPanel} onSelect={setBottomPanel} />
          <PanelButton
            icon={<FlaskConical size={17} />}
            label="Experiments"
            value="experiments"
            selected={bottomPanel}
            onSelect={setBottomPanel}
          />
        </div>

        {bottomPanel === "trace" ? (
          <div className="traceTable">
            <div className="tableHead">Node</div>
            <div className="tableHead">Input</div>
            <div className="tableHead">Output</div>
            <div className="tableHead">Parameters</div>
            {resolution.trace.map((entry) => (
              <Fragment key={entry.nodeId}>
                <div className={entry.errors.length ? "errorText" : undefined}>{entry.name}</div>
                <div>{formatShape(entry.inputShape)}</div>
                <div>{formatShape(entry.outputShape)}</div>
                <div>{formatCompactNumber(entry.parameters)}</div>
              </Fragment>
            ))}
          </div>
        ) : null}

        {bottomPanel === "validation" ? (
          <div className="validationList">
            {resolution.errors.length === 0 ? (
              <div className="validationItem good">
                <CheckCircle2 size={18} />
                <span>Topology passes the current MNIST validation rules.</span>
              </div>
            ) : (
              resolution.errors.map((error) => (
                <div className="validationItem bad" key={error}>
                  <XCircle size={18} />
                  <span>{error}</span>
                </div>
              ))
            )}
          </div>
        ) : null}

        {bottomPanel === "ir" ? <pre className="codePreview">{irPreview}</pre> : null}
        {bottomPanel === "experiments" ? <pre className="codePreview">{experimentPreview}</pre> : null}
      </section>
    </main>
  );
}

function SummaryItem({ label, value, tone }: { label: string; value: string; tone?: "good" | "bad" }) {
  return (
    <div className={`summaryItem ${tone ?? ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PanelButton({
  icon,
  label,
  value,
  selected,
  onSelect,
}: {
  icon: ReactNode;
  label: string;
  value: BottomPanel;
  selected: BottomPanel;
  onSelect: (value: BottomPanel) => void;
}) {
  return (
    <button className={selected === value ? "selected" : ""} type="button" onClick={() => onSelect(value)}>
      {icon}
      {label}
    </button>
  );
}

function buildIntermediatePreview(project: TopologyProject) {
  const stages = project.nodes
    .filter((node) => node.id !== "input")
    .map((node) => {
      const parameters = Object.entries(node.parameters)
        .map(([key, value]) => `      ${key}: ${value}`)
        .join("\n");
      return `  - id: ${node.id}\n    operator: ${node.kind}\n${parameters}`;
    })
    .join("\n");

  return `network:\n  name: ${project.name.toLowerCase().replaceAll(" ", "_")}\n  version: ${project.version}\n  input_shape: ${formatShape(project.inputShape)}\n  stages:\n${stages}`;
}

function buildExperimentPreview(project: TopologyProject) {
  const block5 = project.nodes.find((node) => node.id === "block5");
  const baseWidth = typeof block5?.parameters.out_channels === "number" ? block5.parameters.out_channels : 160;
  const widths = [baseWidth - 16, baseWidth, baseWidth + 16, baseWidth + 32].filter((width) => width > 0);
  const runCount = widths.length * 3 * 3;

  return `experiment_group: fiveblock_pooling_width_search\nbase_network: ${project.name.toLowerCase().replaceAll(" ", "_")}\nexpected_runs: ${runCount}\nsweep:\n  block5.out_channels:\n${widths
    .map((width) => `    - ${width}`)
    .join("\n")}\n  pooling.type:\n    - gap\n    - gap_gmp\n    - gem\nseeds:\n  - 1\n  - 2\n  - 3`;
}
