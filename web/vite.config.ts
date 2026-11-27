import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `base: "./"` lets the bundle be served from /app by FastAPI without
// hard-coding an origin.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { host: "127.0.0.1", port: 5173 },
  build: { outDir: "dist", emptyOutDir: true },
});
