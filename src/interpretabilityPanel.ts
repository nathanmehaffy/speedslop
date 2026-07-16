import { WORLD_SIZE } from "./config";
import type { AgentSample, ClusterSummary, InterpretabilitySnapshot, ProbeAgent, ProbeScenario } from "./interpretabilityTypes";
import { describeBehaviors } from "./behavior";
import { clusterBehaviors, clusterGenomes, type ClusterAssignment } from "./clusters";
import { ancestryForAgent, summarizeLineages } from "./lineage";
import { runProbeScenario, standardProbeScenarios } from "./neuralProbe";

type Tab = "dashboard" | "map" | "lineages" | "clusters" | "agent" | "probe";
type ColorMode = "hue" | "lineage" | "genetic" | "behavior" | "age" | "children";

export class InterpretabilityPanel {
  private snapshot: InterpretabilitySnapshot | null = null;
  private tab: Tab = "dashboard";
  private colorMode: ColorMode = "lineage";
  private selectedAgent: AgentSample | null = null;
  private manualNeighbors: ProbeAgent[] = [];

  constructor(
    private readonly root: HTMLElement,
    private readonly onBack: () => void,
  ) {}

  show(snapshot: InterpretabilitySnapshot): void {
    this.snapshot = snapshot;
    this.selectedAgent = snapshot.agents.find((agent) => agent.alive) ?? null;
    this.manualNeighbors = [];
    this.root.hidden = false;
    this.render();
  }

  hide(): void {
    this.snapshot = null;
    this.root.hidden = true;
    this.root.innerHTML = "";
  }

  private render(): void {
    const snapshot = this.requireSnapshot();
    const previousAgents = snapshot.agentHistory.at(-1) ?? [];
    const descriptors = describeBehaviors(snapshot.agents, snapshot.lifeRecords, snapshot.step, previousAgents);
    const genetic = clusterGenomes(snapshot.genomes);
    const behavior = clusterBehaviors(descriptors);
    const lineages = summarizeLineages(snapshot.agents, snapshot.lifeRecords, snapshot.step);
    const live = snapshot.agents.filter((agent) => agent.alive);
    const selected = this.selectedAgent && this.selectedAgent.alive ? this.selectedAgent : live[0] ?? null;
    this.selectedAgent = selected;

    this.root.innerHTML = `
      <section class="analysis-shell" aria-label="Interpretability mode">
        <header class="analysis-header">
          <div>
            <h1>Interpretability</h1>
            <p>Frozen at step ${formatInt(snapshot.step)} with ${formatInt(live.length)} live agents</p>
          </div>
          <div class="analysis-actions">
            <label>Color
              <select id="analysis-color">
                ${option("lineage", "Lineage", this.colorMode)}
                ${option("hue", "Hue", this.colorMode)}
                ${option("genetic", "Genetic cluster", this.colorMode)}
                ${option("behavior", "Behavior cluster", this.colorMode)}
                ${option("age", "Age", this.colorMode)}
                ${option("children", "Reproductive success", this.colorMode)}
              </select>
            </label>
            <button type="button" id="analysis-back">Back to Simulation</button>
          </div>
        </header>
        <nav class="analysis-tabs" aria-label="Analysis views">
          ${tabButton("dashboard", "Dashboard", this.tab)}
          ${tabButton("map", "World Map", this.tab)}
          ${tabButton("lineages", "Lineages", this.tab)}
          ${tabButton("clusters", "Clusters", this.tab)}
          ${tabButton("agent", "Agent Inspector", this.tab)}
          ${tabButton("probe", "Probe Sandbox", this.tab)}
        </nav>
        <main class="analysis-body">
          ${this.renderActiveTab(snapshot, live, lineages, genetic.summaries, behavior.summaries, genetic.assignments, behavior.assignments)}
        </main>
      </section>
    `;

    this.attachEvents(genetic.assignments, behavior.assignments);
    this.drawCharts(snapshot);
    this.drawMap(genetic.assignments, behavior.assignments);
    this.drawProbe();
  }

  private renderActiveTab(
    snapshot: InterpretabilitySnapshot,
    live: AgentSample[],
    lineages: ReturnType<typeof summarizeLineages>,
    geneticSummaries: ClusterSummary[],
    behaviorSummaries: ClusterSummary[],
    geneticAssignments: ClusterAssignment[],
    behaviorAssignments: ClusterAssignment[],
  ): string {
    if (this.tab === "dashboard") {
      const latest = snapshot.metaSamples.at(-1);
      const previous = snapshot.metaSamples.at(-2);
      const birthsPerSec = rate(latest?.birthTotal, previous?.birthTotal, latest?.recordedAtMs, previous?.recordedAtMs);
      const deathsPerSec = rate(latest?.deathTotal, previous?.deathTotal, latest?.recordedAtMs, previous?.recordedAtMs);
      const immigrantsPerSec = rate(latest?.immigrantTotal, previous?.immigrantTotal, latest?.recordedAtMs, previous?.recordedAtMs);
      const dominant = lineages[0]?.living ?? 0;
      return `
        <div class="analysis-grid">
          ${metric("Live population", formatInt(live.length))}
          ${metric("Births/s", birthsPerSec.toFixed(1))}
          ${metric("Deaths/s", deathsPerSec.toFixed(1))}
          ${metric("Immigrants/s", immigrantsPerSec.toFixed(1))}
          ${metric("Dominant lineage", dominant ? `${formatInt(dominant)} living` : "none")}
          ${metric("Genetic clusters", String(geneticSummaries.length))}
          ${metric("Behavior clusters", String(behaviorSummaries.length))}
          ${metric("Mean death age", latest ? latest.meanDeathAge.toFixed(1) : "0.0")}
        </div>
        <div class="analysis-chart-row">
          <canvas id="population-chart" width="760" height="220" aria-label="Population chart"></canvas>
          <canvas id="event-chart" width="760" height="220" aria-label="Event chart"></canvas>
        </div>
      `;
    }
    if (this.tab === "map") {
      return `<canvas id="analysis-map" width="900" height="620" aria-label="Frozen world map"></canvas>`;
    }
    if (this.tab === "lineages") {
      return `
        <table class="analysis-table">
          <thead><tr><th>Lineage</th><th>Living</th><th>Birth records</th><th>Mean age</th><th>Child count</th></tr></thead>
          <tbody>${lineages.slice(0, 24).map((lineage) => `
            <tr><td>${lineage.lineageId}</td><td>${lineage.living}</td><td>${lineage.births}</td><td>${lineage.meanAge.toFixed(1)}</td><td>${lineage.childCount}</td></tr>
          `).join("")}</tbody>
        </table>
      `;
    }
    if (this.tab === "clusters") {
      const protoEvidence = protoSpeciesEvidence(geneticSummaries, behaviorSummaries);
      return `
        <section class="analysis-split">
          <div>${clusterList("Genetic clusters", geneticSummaries)}</div>
          <div>${clusterList("Behavior clusters", behaviorSummaries)}</div>
        </section>
        <p class="analysis-note">${protoEvidence}</p>
      `;
    }
    if (this.tab === "agent") {
      return this.renderAgentInspector(snapshot, geneticAssignments, behaviorAssignments);
    }
    return this.renderProbeTab(snapshot);
  }

  private renderAgentInspector(
    snapshot: InterpretabilitySnapshot,
    geneticAssignments: readonly ClusterAssignment[],
    behaviorAssignments: readonly ClusterAssignment[],
  ): string {
    const selected = this.selectedAgent;
    if (!selected) {
      return `<p class="analysis-note">No live agent selected.</p>`;
    }
    const life = snapshot.lifeRecords[selected.slot];
    const geneticCluster = geneticAssignments.find((assignment) => assignment.slot === selected.slot)?.clusterId ?? 0;
    const behaviorCluster = behaviorAssignments.find((assignment) => assignment.slot === selected.slot)?.clusterId ?? 0;
    const ancestry = ancestryForAgent(selected, snapshot.agents, snapshot.lifeRecords);
    return `
      <section class="agent-inspector">
        <canvas id="analysis-map" width="640" height="440" aria-label="Agent selection map"></canvas>
        <div class="agent-details">
          ${metric("Agent ID", String(selected.id))}
          ${metric("Slot", String(selected.slot))}
          ${metric("Age", String(Math.max(0, snapshot.step - life.birthStep)))}
          ${metric("Lineage", String(life.lineageId))}
          ${metric("Parents", life.parentAId || life.parentBId ? `${life.parentAId} / ${life.parentBId}` : "founder")}
          ${metric("Children", String(life.childCount))}
          ${metric("Genetic cluster", clusterText(geneticCluster))}
          ${metric("Behavior cluster", clusterText(behaviorCluster))}
          <h2>Ancestry</h2>
          <ol>${ancestry.slice(0, 8).map((record) => `<li>slot ${record.slot}, lineage ${record.lineageId}, born step ${record.birthStep}</li>`).join("")}</ol>
        </div>
      </section>
    `;
  }

  private renderProbeTab(snapshot: InterpretabilitySnapshot): string {
    const selected = this.selectedAgent ?? snapshot.agents.find((agent) => agent.alive);
    if (!selected) {
      return `<p class="analysis-note">No live agent available for probing.</p>`;
    }
    return `
      <section class="probe-layout">
        <div class="probe-controls">
          <select id="probe-scenario">
            <option value="0">Alone</option>
            <option value="1">Neighbor ahead</option>
            <option value="2">Neighbor behind</option>
            <option value="3">Head-on</option>
            <option value="4">Crowd</option>
            <option value="manual">Manual sandbox</option>
          </select>
          <button type="button" id="probe-run">Run Probe</button>
          <button type="button" id="probe-clear">Clear Manual</button>
          <p class="analysis-note">Click the sandbox to add manual neighbors around the selected genome.</p>
        </div>
        <canvas id="probe-canvas" width="760" height="520" aria-label="Probe sandbox"></canvas>
        <pre id="probe-output"></pre>
      </section>
    `;
  }

  private attachEvents(geneticAssignments: readonly ClusterAssignment[], behaviorAssignments: readonly ClusterAssignment[]): void {
    this.root.querySelector("#analysis-back")?.addEventListener("click", this.onBack);
    this.root.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        this.tab = button.dataset.tab as Tab;
        this.render();
      });
    });
    this.root.querySelector<HTMLSelectElement>("#analysis-color")?.addEventListener("change", (event) => {
      this.colorMode = (event.currentTarget as HTMLSelectElement).value as ColorMode;
      this.drawMap(geneticAssignments, behaviorAssignments);
    });
    this.root.querySelector<HTMLCanvasElement>("#analysis-map")?.addEventListener("click", (event) => {
      this.selectAgentAt(event.currentTarget as HTMLCanvasElement, event.offsetX, event.offsetY);
      this.render();
    });
    this.root.querySelector("#probe-run")?.addEventListener("click", () => this.runSelectedProbe());
    this.root.querySelector("#probe-clear")?.addEventListener("click", () => {
      this.manualNeighbors = [];
      this.drawProbe();
    });
    this.root.querySelector<HTMLCanvasElement>("#probe-canvas")?.addEventListener("click", (event) => {
      this.addManualNeighbor(event.currentTarget as HTMLCanvasElement, event.offsetX, event.offsetY);
      this.drawProbe();
    });
  }

  private drawCharts(snapshot: InterpretabilitySnapshot): void {
    drawLineChart(this.root.querySelector<HTMLCanvasElement>("#population-chart"), snapshot.metaSamples.map((sample) => sample.liveCount), "#9ecbff", "Population");
    drawLineChart(this.root.querySelector<HTMLCanvasElement>("#event-chart"), snapshot.metaSamples.map((sample) => sample.birthTotal - sample.deathTotal), "#8ee6a8", "Births minus deaths");
  }

  private drawMap(
    geneticAssignments: readonly ClusterAssignment[] = [],
    behaviorAssignments: readonly ClusterAssignment[] = [],
  ): void {
    const canvas = this.root.querySelector<HTMLCanvasElement>("#analysis-map");
    const snapshot = this.snapshot;
    if (!canvas || !snapshot) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    const geneticBySlot = new Map(geneticAssignments.map((assignment) => [assignment.slot, assignment.clusterId]));
    const behaviorBySlot = new Map(behaviorAssignments.map((assignment) => [assignment.slot, assignment.clusterId]));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#050810";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
    const live = snapshot.agents.filter((agent) => agent.alive);
    const maxAge = Math.max(1, ...live.map((agent) => snapshot.step - snapshot.lifeRecords[agent.slot].birthStep));
    const maxChildren = Math.max(1, ...live.map((agent) => snapshot.lifeRecords[agent.slot].childCount));
    for (const agent of live) {
      ctx.fillStyle = colorForAgent(agent, snapshot, this.colorMode, geneticBySlot, behaviorBySlot, maxAge, maxChildren);
      const x = (agent.x / WORLD_SIZE) * canvas.width;
      const y = canvas.height - (agent.y / WORLD_SIZE) * canvas.height;
      ctx.beginPath();
      ctx.arc(x, y, agent.id === this.selectedAgent?.id ? 4 : 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  private selectAgentAt(canvas: HTMLCanvasElement, sx: number, sy: number): void {
    const snapshot = this.snapshot;
    if (!snapshot) {
      return;
    }
    let best: AgentSample | null = null;
    let bestDist = Infinity;
    for (const agent of snapshot.agents) {
      if (!agent.alive) {
        continue;
      }
      const x = (agent.x / WORLD_SIZE) * canvas.width;
      const y = canvas.height - (agent.y / WORLD_SIZE) * canvas.height;
      const d = Math.hypot(sx - x, sy - y);
      if (d < bestDist) {
        best = agent;
        bestDist = d;
      }
    }
    if (best && bestDist <= 16) {
      this.selectedAgent = best;
      this.tab = "agent";
    }
  }

  private runSelectedProbe(): void {
    const scenario = this.selectedProbeScenario();
    if (!scenario) {
      return;
    }
    const trace = runProbeScenario(scenario);
    this.drawProbe(trace.positions);
    const last = trace.neural.at(-1);
    const output = this.root.querySelector<HTMLPreElement>("#probe-output");
    if (output && last) {
      output.textContent = [
        `Scenario: ${trace.scenarioName}`,
        `steps: ${trace.neural.length}`,
        `last turn raw: ${last.turnRaw.toFixed(4)}`,
        `last speed raw: ${last.speedRaw.toFixed(4)}`,
        `last velocity: ${last.nextVel.toFixed(6)}`,
        `collision states: ${summarizeCollisionKinds(trace.collisionKinds)}`,
        `inputs: ${last.inputs.map((value) => value.toFixed(2)).join(", ")}`,
      ].join("\n");
    }
  }

  private selectedProbeScenario(): ProbeScenario | null {
    const snapshot = this.snapshot;
    const selected = this.selectedAgent;
    if (!snapshot || !selected) {
      return null;
    }
    const genome = snapshot.genomes.find((sample) => sample.slot === selected.slot)?.weights;
    if (!genome) {
      return null;
    }
    const focal = agentToProbe(selected, genome);
    const scenarioValue = this.root.querySelector<HTMLSelectElement>("#probe-scenario")?.value ?? "0";
    if (scenarioValue === "manual") {
      return { name: "Manual sandbox", focal, neighbors: this.manualNeighbors, steps: 80 };
    }
    return standardProbeScenarios(focal)[Number(scenarioValue)] ?? null;
  }

  private addManualNeighbor(canvas: HTMLCanvasElement, sx: number, sy: number): void {
    const selected = this.selectedAgent;
    const snapshot = this.snapshot;
    if (!selected || !snapshot) {
      return;
    }
    const genome = snapshot.genomes.find((sample) => sample.slot === selected.slot)?.weights;
    if (!genome) {
      return;
    }
    this.manualNeighbors.push({
      id: 800_000 + this.manualNeighbors.length,
      x: (sx / canvas.width) * WORLD_SIZE,
      y: ((canvas.height - sy) / canvas.height) * WORLD_SIZE,
      dir: Math.PI,
      vel: selected.vel,
      genome,
    });
  }

  private drawProbe(path: readonly { x: number; y: number }[] = []): void {
    const canvas = this.root.querySelector<HTMLCanvasElement>("#probe-canvas");
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.fillStyle = "#050810";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
    for (const neighbor of this.manualNeighbors) {
      ctx.fillStyle = "#ffcf70";
      ctx.beginPath();
      ctx.arc((neighbor.x / WORLD_SIZE) * canvas.width, canvas.height - (neighbor.y / WORLD_SIZE) * canvas.height, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    if (path.length > 0) {
      ctx.strokeStyle = "#9ecbff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      path.forEach((point, index) => {
        const x = (point.x / WORLD_SIZE) * canvas.width;
        const y = canvas.height - (point.y / WORLD_SIZE) * canvas.height;
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }
  }

  private requireSnapshot(): InterpretabilitySnapshot {
    if (!this.snapshot) {
      throw new Error("analysis panel has no snapshot");
    }
    return this.snapshot;
  }
}

function tabButton(tab: Tab, label: string, active: Tab): string {
  return `<button type="button" data-tab="${tab}" aria-pressed="${tab === active}">${label}</button>`;
}

function option(value: ColorMode, label: string, active: ColorMode): string {
  return `<option value="${value}"${value === active ? " selected" : ""}>${label}</option>`;
}

function metric(label: string, value: string): string {
  return `<div class="analysis-metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function clusterList(title: string, summaries: readonly ClusterSummary[]): string {
  return `
    <h2>${title}</h2>
    <table class="analysis-table">
      <thead><tr><th>ID</th><th>Size</th><th>Centroid slot</th><th>Mean distance</th></tr></thead>
      <tbody>${summaries.slice(0, 20).map((cluster) => `
        <tr><td>${cluster.id}</td><td>${cluster.size}</td><td>${cluster.centroidSlot}</td><td>${cluster.meanDistance.toFixed(3)}</td></tr>
      `).join("")}</tbody>
    </table>
  `;
}

function drawLineChart(canvas: HTMLCanvasElement | null, values: readonly number[], color: string, label: string): void {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.fillStyle = "#08101d";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#c8d4e8";
  ctx.font = "12px system-ui";
  ctx.fillText(label, 12, 20);
  if (values.length < 2) {
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = 12 + (index / (values.length - 1)) * (canvas.width - 24);
    const y = canvas.height - 18 - ((value - min) / span) * (canvas.height - 42);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function colorForAgent(
  agent: AgentSample,
  snapshot: InterpretabilitySnapshot,
  mode: ColorMode,
  geneticBySlot: ReadonlyMap<number, number>,
  behaviorBySlot: ReadonlyMap<number, number>,
  maxAge: number,
  maxChildren: number,
): string {
  const life = snapshot.lifeRecords[agent.slot];
  if (mode === "hue") {
    return hsv(agent.hue, agent.sat, agent.val);
  }
  if (mode === "genetic") {
    return hashColor(geneticBySlot.get(agent.slot) ?? 0);
  }
  if (mode === "behavior") {
    return hashColor(behaviorBySlot.get(agent.slot) ?? 0);
  }
  if (mode === "age") {
    return heatColor((snapshot.step - life.birthStep) / maxAge);
  }
  if (mode === "children") {
    return heatColor(life.childCount / maxChildren);
  }
  return hashColor(life.lineageId);
}

function hsv(h: number, s: number, v: number): string {
  return `hsl(${Math.round(h * 360)}, ${Math.round(s * 100)}%, ${Math.round(v * 55)}%)`;
}

function hashColor(value: number): string {
  if (value === 0) {
    return "#788496";
  }
  const hue = (Math.imul(value, 2654435761) >>> 0) % 360;
  return `hsl(${hue}, 76%, 62%)`;
}

function heatColor(value: number): string {
  const hue = 210 - Math.round(210 * Math.max(0, Math.min(1, value)));
  return `hsl(${hue}, 82%, 62%)`;
}

function protoSpeciesEvidence(genetic: readonly ClusterSummary[], behavior: readonly ClusterSummary[]): string {
  if (genetic.length >= 2 && behavior.length >= 2) {
    return "Proto-species evidence: multiple genetic and behavior clusters are visible in this snapshot. Persistence requires at least 3 genome samples before stronger language is shown.";
  }
  return "Proto-species evidence: insufficient. The app is showing cautious clusters only; no cluster has enough persistence/divergence evidence yet.";
}

function rate(current = 0, previous = 0, currentAt = 0, previousAt = 0): number {
  const seconds = (currentAt - previousAt) / 1000;
  return seconds > 0 ? (current - previous) / seconds : 0;
}

function clusterText(clusterId: number): string {
  return clusterId > 0 ? String(clusterId) : "unclustered";
}

function agentToProbe(agent: AgentSample, genome: Float32Array): ProbeAgent {
  return {
    id: agent.id,
    x: WORLD_SIZE / 2,
    y: WORLD_SIZE / 2,
    dir: agent.dir,
    vel: agent.vel,
    genome,
  };
}

function summarizeCollisionKinds(kinds: readonly string[]): string {
  const counts = new Map<string, number>();
  for (const kind of kinds) {
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
  }
  return [...counts.entries()].map(([kind, count]) => `${kind} ${count}`).join(", ");
}

function formatInt(value: number): string {
  return Math.round(value).toLocaleString("en-US");
}
