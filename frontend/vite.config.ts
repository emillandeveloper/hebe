import { defineConfig } from "vite";
import devRuntimeConfig from "./electron/dev_runtime_config.cjs";

export default defineConfig({
  server: {
    host: devRuntimeConfig.DEV_SERVER_HOST,
    port: devRuntimeConfig.DEV_SERVER_PORT,
    strictPort: true,
  },
});
