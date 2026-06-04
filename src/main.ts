import "./style.css";
import { startApp } from "./app";

async function main(): Promise<void> {
  const canvasElement = document.querySelector<HTMLCanvasElement>("#canvas");
  if (!canvasElement) {
    throw new Error("Canvas element #canvas not found");
  }
  const monitor = document.querySelector<HTMLElement>("#fps-monitor");
  const controls = document.querySelector<HTMLElement>("#sim-controls");
  const errorPanel = document.querySelector<HTMLElement>("#error-panel");

  await startApp(
    { canvas: canvasElement, monitor, controls },
    {
      onFatalError(error) {
        showFatalError(errorPanel, error);
      },
    },
  );
}

function showFatalError(errorPanel: HTMLElement | null, error: unknown): void {
  console.error(error);
  const message = error instanceof Error ? error.message : String(error);
  if (errorPanel) {
    errorPanel.hidden = false;
    errorPanel.textContent = message;
    return;
  }
  document.body.textContent = `SpeedSlop can't start: ${message}`;
}

main().catch((error: unknown) => {
  const errorPanel = document.querySelector<HTMLElement>("#error-panel");
  showFatalError(errorPanel, error);
});
