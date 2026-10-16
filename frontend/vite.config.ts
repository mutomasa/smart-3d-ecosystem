import { defineConfig } from "vitest/config";

/** Backend used by the dev proxy; override when port 8000 is taken (#14). */
const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";
const BACKEND_WS = BACKEND_URL.replace(/^http/, "ws");

export default defineConfig({
  base: "./",
  server: {
    port: Number(process.env.FRONTEND_PORT ?? 5173),
    fs: { allow: [".."] },
    proxy: {
      "/api": BACKEND_URL,
      "/health": BACKEND_URL,
      "/ws": { target: BACKEND_WS, ws: true },
    },
  },
  worker: { format: "es" },
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
