import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const RESULT_MARKER = "SPEEDSLOP_SIM_BENCH_RESULT";
const ERROR_MARKER = "SPEEDSLOP_SIM_BENCH_ERROR";
const DEFAULT_TIMEOUT_MS = 60_000;

const runnerOptions = new Set(["chrome", "timeout-ms", "json"]);
const benchOptions = new Set(["population", "steps", "warmup", "batch", "seed", "world-size"]);

function parseArgs(argv) {
  const options = {
    bench: new Map(),
    chrome: null,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    json: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected argument: ${arg}`);
    }

    const withoutPrefix = arg.slice(2);
    const equalsIndex = withoutPrefix.indexOf("=");
    const key = equalsIndex === -1 ? withoutPrefix : withoutPrefix.slice(0, equalsIndex);
    let value = equalsIndex === -1 ? null : withoutPrefix.slice(equalsIndex + 1);

    if (!runnerOptions.has(key) && !benchOptions.has(key)) {
      throw new Error(`Unknown option: --${key}`);
    }

    if (key === "json") {
      if (value !== null) {
        throw new Error("--json does not accept a value.");
      }
      options.json = true;
      continue;
    }

    if (value === null) {
      index += 1;
      value = argv[index] ?? null;
    }
    if (value === null || value.startsWith("--")) {
      throw new Error(`Missing value for --${key}.`);
    }

    if (key === "chrome") {
      options.chrome = value;
    } else if (key === "timeout-ms") {
      options.timeoutMs = parsePositiveInteger(value, key);
    } else {
      options.bench.set(key, String(parseNonNegativeInteger(value, key)));
    }
  }

  return options;
}

function parsePositiveInteger(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    throw new Error(`Expected --${name} to be a positive number.`);
  }
  return Math.floor(number);
}

function parseNonNegativeInteger(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    throw new Error(`Expected --${name} to be zero or a positive number.`);
  }
  return Math.floor(number);
}

function chromeCandidates() {
  const localAppData = process.env.LOCALAPPDATA;
  const programFiles = process.env.ProgramFiles;
  const programFilesX86 = process.env["ProgramFiles(x86)"];

  return [
    programFiles && path.join(programFiles, "Google", "Chrome", "Application", "chrome.exe"),
    programFilesX86 && path.join(programFilesX86, "Google", "Chrome", "Application", "chrome.exe"),
    localAppData && path.join(localAppData, "Google", "Chrome", "Application", "chrome.exe"),
    programFiles && path.join(programFiles, "Microsoft", "Edge", "Application", "msedge.exe"),
    programFilesX86 && path.join(programFilesX86, "Microsoft", "Edge", "Application", "msedge.exe"),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
}

function findChrome(override) {
  if (override) {
    if (!existsSync(override)) {
      throw new Error(`Chrome executable does not exist: ${override}`);
    }
    return override;
  }

  const found = chromeCandidates().find((candidate) => existsSync(candidate));
  if (!found) {
    throw new Error("Could not find Chrome or Edge. Pass --chrome=<path> to choose an executable.");
  }

  return found;
}

async function waitForDevToolsPort(profileDir, child, timeoutMs) {
  const activePortPath = path.join(profileDir, "DevToolsActivePort");
  const startedAt = Date.now();
  let stderr = "";

  child.stderr?.on("data", (chunk) => {
    stderr += chunk.toString();
  });

  while (Date.now() - startedAt < timeoutMs) {
    if (child.exitCode !== null) {
      throw new Error(`Chrome exited before DevTools became available.\n${stderr.trim()}`);
    }

    try {
      const content = await readFile(activePortPath, "utf8");
      const [portLine] = content.trim().split(/\r?\n/);
      const port = Number(portLine);
      if (Number.isInteger(port) && port > 0) {
        return port;
      }
    } catch {
      // Chrome creates DevToolsActivePort asynchronously.
    }

    await delay(50);
  }

  throw new Error(`Timed out waiting for Chrome DevTools port.\n${stderr.trim()}`);
}

async function createTarget(port, url) {
  const endpoint = `http://127.0.0.1:${port}/json/new?${encodeURIComponent(url)}`;
  const response = await fetch(endpoint, { method: "PUT" });
  if (!response.ok) {
    throw new Error(`Failed to create Chrome target: ${response.status} ${response.statusText}`);
  }

  const target = await response.json();
  if (typeof target.webSocketDebuggerUrl !== "string") {
    throw new Error("Chrome target did not include a DevTools websocket URL.");
  }

  return target.webSocketDebuggerUrl;
}

class CdpClient {
  constructor(url) {
    this.url = url;
    this.socket = null;
    this.nextId = 1;
    this.pending = new Map();
    this.handlers = new Map();
  }

  async connect() {
    this.socket = new WebSocket(this.url);
    this.socket.addEventListener("message", (event) => this.handleMessage(event.data));
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
  }

  send(method, params = {}) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      throw new Error("DevTools websocket is not open.");
    }

    const id = this.nextId;
    this.nextId += 1;

    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });

    this.socket.send(JSON.stringify({ id, method, params }));
    return promise;
  }

  on(method, handler) {
    const handlers = this.handlers.get(method) ?? [];
    handlers.push(handler);
    this.handlers.set(method, handlers);
  }

  close() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
    }
  }

  handleMessage(raw) {
    const text = typeof raw === "string" ? raw : Buffer.from(raw).toString("utf8");
    const message = JSON.parse(text);

    if (typeof message.id === "number") {
      const pending = this.pending.get(message.id);
      if (!pending) {
        return;
      }

      this.pending.delete(message.id);
      if (message.error) {
        pending.reject(new Error(`${message.error.message}: ${message.error.data ?? ""}`.trim()));
      } else {
        pending.resolve(message.result);
      }
      return;
    }

    if (typeof message.method === "string") {
      for (const handler of this.handlers.get(message.method) ?? []) {
        handler(message.params ?? {});
      }
    }
  }
}

function consoleText(params) {
  return (params.args ?? [])
    .map((arg) => {
      if (typeof arg.value === "string") {
        return arg.value;
      }
      if (arg.value !== undefined) {
        return String(arg.value);
      }
      return arg.description ?? "";
    })
    .join(" ");
}

function buildBenchmarkUrl(baseUrl, benchOptionsMap) {
  const url = new URL("/sim-bench.html", baseUrl);
  for (const [key, value] of benchOptionsMap) {
    url.searchParams.set(key, value);
  }
  return url.href;
}

function formatAdapter(adapter) {
  if (!adapter || typeof adapter !== "object") {
    return "unknown";
  }

  const details = [adapter.vendor, adapter.architecture, adapter.device, adapter.description].filter(Boolean);
  return details.length > 0 ? details.join(" / ") : "unknown";
}

function printHumanResult(result, chromePath) {
  console.log("SpeedSlop headless simulation benchmark");
  console.log(`Chrome: ${chromePath}`);
  console.log(`Adapter: ${formatAdapter(result.adapter)}`);
  console.log(`Population: ${result.population.toLocaleString()} agents`);
  console.log(`World size: ${result.worldSize.toLocaleString()}`);
  console.log(`Seed: ${result.seed.toLocaleString()}`);
  console.log(`Warmup: ${result.warmupSteps.toLocaleString()} steps`);
  console.log(`Measured: ${result.measuredSteps.toLocaleString()} steps`);
  console.log(`Batch size: ${result.batchSize.toLocaleString()} steps`);
  console.log(`Elapsed: ${result.elapsedMs.toFixed(1)} ms`);
  console.log(`Simulation: ${result.stepsPerSecond.toLocaleString()} steps/s`);
  console.log(`Agent updates: ${result.agentUpdatesPerSecond.toLocaleString()} agents/s`);
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForExit(child) {
  if (child.exitCode !== null) {
    return;
  }

  await new Promise((resolve) => {
    child.once("exit", resolve);
  });
}

async function main() {
  const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const options = parseArgs(process.argv.slice(2));
  const chromePath = findChrome(options.chrome);
  const profileDir = await mkdtemp(path.join(tmpdir(), "speedslop-chrome-"));
  const server = await createServer({
    root: projectRoot,
    logLevel: "error",
    server: {
      host: "127.0.0.1",
      port: 0,
      strictPort: false,
    },
  });
  let chrome = null;
  let cdp = null;

  try {
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === "string") {
      throw new Error("Unable to determine Vite server port.");
    }

    const baseUrl = `http://127.0.0.1:${address.port}`;
    const benchmarkUrl = buildBenchmarkUrl(baseUrl, options.bench);
    chrome = spawn(
      chromePath,
      [
        "--headless=new",
        "--remote-debugging-port=0",
        `--user-data-dir=${profileDir}`,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--enable-unsafe-webgpu",
        "--enable-webgpu-developer-features",
        "--ignore-gpu-blocklist",
        "about:blank",
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );

    const port = await waitForDevToolsPort(profileDir, chrome, options.timeoutMs);
    const targetWebSocketUrl = await createTarget(port, benchmarkUrl);
    cdp = new CdpClient(targetWebSocketUrl);
    await cdp.connect();

    const browserLogs = [];
    const resultPromise = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Timed out after ${options.timeoutMs} ms waiting for benchmark result.`));
      }, options.timeoutMs);

      cdp.on("Runtime.consoleAPICalled", (params) => {
        const text = consoleText(params);
        browserLogs.push(text);

        if (text.startsWith(RESULT_MARKER)) {
          clearTimeout(timeout);
          resolve(JSON.parse(text.slice(RESULT_MARKER.length).trim()));
        } else if (text.startsWith(ERROR_MARKER)) {
          clearTimeout(timeout);
          const payload = JSON.parse(text.slice(ERROR_MARKER.length).trim());
          reject(new Error(payload.message ?? "Benchmark failed in browser."));
        }
      });

      cdp.on("Runtime.exceptionThrown", (params) => {
        const description = params.exceptionDetails?.exception?.description ?? params.exceptionDetails?.text ?? "Browser exception";
        clearTimeout(timeout);
        reject(new Error(description));
      });
    });

    await cdp.send("Runtime.enable");
    await cdp.send("Page.enable");
    await cdp.send("Page.navigate", { url: benchmarkUrl });

    const result = await resultPromise.catch((error) => {
      if (browserLogs.length > 0) {
        error.message = `${error.message}\nBrowser console:\n${browserLogs.join("\n")}`;
      }
      throw error;
    });

    if (options.json) {
      console.log(JSON.stringify({ chrome: chromePath, ...result }));
    } else {
      printHumanResult(result, chromePath);
    }
  } finally {
    cdp?.close();
    if (chrome && chrome.exitCode === null) {
      chrome.kill();
      await waitForExit(chrome);
    }
    await server.close();
    await rm(profileDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
