import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { chromium } from "playwright";
import { createServer } from "vite";

const DEFAULT_OPTIONS = {
  samples: 25,
  warmup: 5,
  width: 1280,
  height: 720,
  stepBatches: [1, 4, 16, 64],
};

const args = parseArgs(process.argv.slice(2));
const options = {
  samples: args.samples ?? DEFAULT_OPTIONS.samples,
  warmup: args.warmup ?? DEFAULT_OPTIONS.warmup,
  width: args.width ?? DEFAULT_OPTIONS.width,
  height: args.height ?? DEFAULT_OPTIONS.height,
  stepBatches: args.stepBatches ?? DEFAULT_OPTIONS.stepBatches,
  profileStages: args.profileStages ?? false,
};

const server = await createServer({
  configFile: resolve("vite.config.ts"),
  server: { host: "127.0.0.1" },
  logLevel: "error",
});

let browser;
try {
  await server.listen();
  const url = `${server.resolvedUrls.local[0]}benchmark.html`;
  browser = await launchBrowser(args.headed);
  const page = await browser.newPage({
    viewport: { width: options.width, height: options.height },
    deviceScaleFactor: 1,
  });

  page.setDefaultTimeout(120_000);
  page.on("console", (message) => {
    if (message.type() === "error") {
      console.error(`[browser] ${message.text()}`);
    }
  });

  await page.goto(url);
  await page.waitForFunction(() => typeof window.speedSlopRunBenchmark === "function");
  const report = await page.evaluate((browserOptions) => window.speedSlopRunBenchmark(browserOptions), options);

  printReport(report);
  if (args.jsonPath) {
    const outputPath = resolve(args.jsonPath);
    await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    console.log(`\nWrote JSON report to ${outputPath}`);
  }
} catch (error) {
  console.error("\nGPU benchmark failed.");
  console.error(error instanceof Error ? error.message : String(error));
  console.error("\nThis benchmark requires a browser with WebGPU and the timestamp-query feature.");
  console.error("If Playwright has no browser installed, run: npx playwright install chromium");
  process.exitCode = 1;
} finally {
  await browser?.close();
  await server.close();
}

async function launchBrowser(headed) {
  const launchOptions = {
    headless: !headed,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--disable-background-timer-throttling",
      "--disable-renderer-backgrounding",
    ],
  };
  try {
    return await chromium.launch(launchOptions);
  } catch (firstError) {
    for (const channel of ["msedge", "chrome"]) {
      try {
        return await chromium.launch({ ...launchOptions, channel });
      } catch {
        // Try the next installed browser channel.
      }
    }
    throw firstError;
  }
}

function printReport(report) {
  console.log("SpeedSlop GPU benchmark");
  console.log(`Created: ${report.createdAt}`);
  console.log(`User agent: ${report.environment.userAgent}`);
  console.log(`Canvas: ${report.environment.canvasWidth}x${report.environment.canvasHeight}`);
  console.log(`Agents: ${report.environment.maxAgents}, grid: ${report.environment.gridDim}x${report.environment.gridDim}`);
  console.log("");
  console.log([
    "case",
    "workload",
    "gpu median",
    "gpu p95",
    "cpu encode median",
    "steps/sec",
  ].join("\t"));
  for (const entry of report.cases) {
    const workload = Object.entries(entry.workload)
      .map(([key, value]) => `${key}=${value}`)
      .join(",");
    console.log([
      entry.name,
      workload,
      `${entry.gpuMs.median.toFixed(3)} ms`,
      `${entry.gpuMs.p95.toFixed(3)} ms`,
      `${entry.cpuEncodeMs.median.toFixed(3)} ms`,
      entry.stepsPerSecond ? Math.round(entry.stepsPerSecond).toString() : "-",
    ].join("\t"));
  }
}

function parseArgs(rawArgs) {
  const parsed = {
    headed: false,
    jsonPath: "benchmark-results.json",
  };
  for (let i = 0; i < rawArgs.length; i += 1) {
    const arg = rawArgs[i];
    const [flag, inlineValue] = arg.split("=", 2);
    const value = inlineValue ?? rawArgs[i + 1];
    if (inlineValue === undefined && flag !== "--headed" && flag !== "--no-json" && flag !== "--profile-stages") {
      i += 1;
    }

    switch (flag) {
      case "--samples":
        parsed.samples = parsePositiveInteger(value, flag);
        break;
      case "--warmup":
        parsed.warmup = parseNonNegativeInteger(value, flag);
        break;
      case "--width":
        parsed.width = parsePositiveInteger(value, flag);
        break;
      case "--height":
        parsed.height = parsePositiveInteger(value, flag);
        break;
      case "--steps":
        parsed.stepBatches = value.split(",").map((item) => parsePositiveInteger(item, flag));
        break;
      case "--json":
        parsed.jsonPath = value;
        break;
      case "--no-json":
        parsed.jsonPath = null;
        break;
      case "--headed":
        parsed.headed = true;
        break;
      case "--profile-stages":
        parsed.profileStages = true;
        break;
      default:
        throw new Error(`Unknown benchmark option: ${flag}`);
    }
  }
  return parsed;
}

function parsePositiveInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer`);
  }
  return parsed;
}

function parseNonNegativeInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative integer`);
  }
  return parsed;
}
