// Pure 2D camera: pan, zoom-about-cursor, world<->screen mapping, and the set
// of torus tiles a viewport currently covers. No DOM or GPU dependencies, so it
// is fully unit-testable.

import { MAX_ZOOM, MIN_ZOOM, WORLD_SIZE, ZOOM_SENSITIVITY } from "./config";

export interface Viewport {
  width: number;
  height: number;
}

export interface Vec2 {
  x: number;
  y: number;
}

/** Inclusive integer tile range covering the viewport (torus wrapping). */
export interface TileRange {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export class Camera {
  /** World-space point at the centre of the viewport. */
  center: Vec2;
  /** Pixels per world unit. */
  zoom: number;

  constructor(center: Vec2 = { x: WORLD_SIZE / 2, y: WORLD_SIZE / 2 }, zoom: number = MIN_ZOOM) {
    this.center = { x: center.x, y: center.y };
    this.zoom = clampZoom(zoom);
  }

  /** Set zoom so the unit world fills `fraction` of the smaller viewport axis. */
  fitWorld(viewport: Viewport, fraction = 0.9): void {
    const minAxis = Math.min(viewport.width, viewport.height);
    this.zoom = clampZoom((minAxis * fraction) / WORLD_SIZE);
    this.center = { x: WORLD_SIZE / 2, y: WORLD_SIZE / 2 };
  }

  /** Pan by a screen-space drag delta (pixels). */
  pan(dxPixels: number, dyPixels: number): void {
    this.center.x -= dxPixels / this.zoom;
    this.center.y += dyPixels / this.zoom; // screen y is down, world y is up
  }

  /** Zoom by a wheel delta, keeping the world point under the cursor fixed. */
  zoomBy(wheelDeltaY: number, cursor: Vec2, viewport: Viewport): void {
    const before = this.screenToWorld(cursor, viewport);
    this.zoom = clampZoom(this.zoom * Math.exp(-wheelDeltaY * ZOOM_SENSITIVITY));
    const halfW = viewport.width / 2;
    const halfH = viewport.height / 2;
    this.center.x = before.x - (cursor.x - halfW) / this.zoom;
    this.center.y = before.y + (cursor.y - halfH) / this.zoom;
  }

  worldToScreen(world: Vec2, viewport: Viewport): Vec2 {
    return {
      x: (world.x - this.center.x) * this.zoom + viewport.width / 2,
      y: viewport.height / 2 - (world.y - this.center.y) * this.zoom,
    };
  }

  screenToWorld(screen: Vec2, viewport: Viewport): Vec2 {
    return {
      x: (screen.x - viewport.width / 2) / this.zoom + this.center.x,
      y: this.center.y - (screen.y - viewport.height / 2) / this.zoom,
    };
  }

  /** Integer tile indices whose copy of the world overlaps the viewport. */
  visibleTiles(viewport: Viewport): TileRange {
    const halfWorldW = viewport.width / 2 / this.zoom;
    const halfWorldH = viewport.height / 2 / this.zoom;
    const left = this.center.x - halfWorldW;
    const right = this.center.x + halfWorldW;
    const bottom = this.center.y - halfWorldH;
    const top = this.center.y + halfWorldH;
    return {
      minX: Math.floor(left / WORLD_SIZE),
      maxX: Math.floor(right / WORLD_SIZE),
      minY: Math.floor(bottom / WORLD_SIZE),
      maxY: Math.floor(top / WORLD_SIZE),
    };
  }
}

function clampZoom(zoom: number): number {
  if (zoom < MIN_ZOOM) {
    return MIN_ZOOM;
  }
  if (zoom > MAX_ZOOM) {
    return MAX_ZOOM;
  }
  return zoom;
}
