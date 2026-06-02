import { describe, expect, it } from "vitest";

import { Camera, MAX_VISIBLE_TILES, clampTileRange, tileOffsetsForRange, type Viewport } from "./camera";
import { WORLD_SIZE, ZOOM_IN_LIMIT, ZOOM_OUT_LIMIT } from "./config";

const viewport: Viewport = { width: 800, height: 600 };

describe("Camera world<->screen", () => {
  it("places the camera centre at the viewport centre", () => {
    const cam = new Camera({ x: 0.5, y: 0.5 }, 100);
    const screen = cam.worldToScreen({ x: 0.5, y: 0.5 }, viewport);
    expect(screen.x).toBeCloseTo(400, 9);
    expect(screen.y).toBeCloseTo(300, 9);
  });

  it("round-trips screen->world->screen", () => {
    const cam = new Camera({ x: 0.3, y: 0.7 }, 250);
    const world = cam.screenToWorld({ x: 123, y: 456 }, viewport);
    const screen = cam.worldToScreen(world, viewport);
    expect(screen.x).toBeCloseTo(123, 6);
    expect(screen.y).toBeCloseTo(456, 6);
  });
});

describe("Camera pan", () => {
  it("moves the world under the cursor opposite to the drag", () => {
    const cam = new Camera({ x: 0.5, y: 0.5 }, 100);
    cam.pan(100, 0); // drag right by 100px
    expect(cam.center.x).toBeCloseTo(0.5 - 100 / 100, 9);
  });

  it("inverts screen-y so dragging down moves the view down", () => {
    const cam = new Camera({ x: 0.5, y: 0.5 }, 100);
    cam.pan(0, 50);
    expect(cam.center.y).toBeCloseTo(0.5 + 50 / 100, 9);
  });
});

describe("Camera zoomBy", () => {
  it("keeps the world point under the cursor fixed", () => {
    const cam = new Camera({ x: 0.5, y: 0.5 }, 200);
    const cursor = { x: 600, y: 200 };
    const before = cam.screenToWorld(cursor, viewport);
    cam.zoomBy(-240, cursor, viewport); // zoom in
    const after = cam.screenToWorld(cursor, viewport);
    expect(after.x).toBeCloseTo(before.x, 6);
    expect(after.y).toBeCloseTo(before.y, 6);
    expect(cam.zoom).toBeGreaterThan(200);
  });

  it("clamps zoom to limits relative to the fit-world reference", () => {
    const cam = new Camera();
    cam.fitWorld(viewport);
    const reference = (Math.min(viewport.width, viewport.height) * 0.9) / WORLD_SIZE;

    cam.zoomBy(-1e6, { x: 400, y: 300 }, viewport);
    expect(cam.zoom).toBeCloseTo(reference * ZOOM_IN_LIMIT, 9);
    cam.zoomBy(1e6, { x: 400, y: 300 }, viewport);
    expect(cam.zoom).toBeCloseTo(reference / ZOOM_OUT_LIMIT, 9);
  });
});

describe("Camera visibleTiles", () => {
  it("returns a single tile when fully zoomed onto one copy", () => {
    const cam = new Camera({ x: 0.5, y: 0.5 }, 100_000);
    const tiles = cam.visibleTiles(viewport);
    expect(tiles).toEqual({ minX: 0, maxX: 0, minY: 0, maxY: 0 });
  });

  it("spans negative and positive tiles when zoomed out", () => {
    const mid = WORLD_SIZE / 2;
    const cam = new Camera({ x: mid, y: mid }, 100);
    const tiles = cam.visibleTiles(viewport);
    expect(tiles.minX).toBeLessThan(0);
    expect(tiles.maxX).toBeGreaterThan(0);
    expect(tiles.minY).toBeLessThan(0);
    expect(tiles.maxY).toBeGreaterThan(0);
  });
});

describe("camera tile budgeting", () => {
  it("clamps a large tile range to the max tile budget", () => {
    const tiles = clampTileRange({ minX: -100, maxX: 100, minY: -100, maxY: 100 });
    const width = tiles.maxX - tiles.minX + 1;
    const height = tiles.maxY - tiles.minY + 1;

    expect(width * height).toBeLessThanOrEqual(MAX_VISIBLE_TILES);
  });

  it("returns world-space offsets for the clamped tile range", () => {
    const offsets = tileOffsetsForRange({ minX: -1, maxX: 1, minY: 0, maxY: 0 });

    expect(offsets).toEqual([
      [-WORLD_SIZE, 0],
      [0, 0],
      [WORLD_SIZE, 0],
    ]);
  });
});
