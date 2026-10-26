import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const directory = fileURLToPath(new URL("../public/", import.meta.url));
mkdirSync(directory, { recursive: true });
writeFileSync(
  directory + "snapshot-assets.json",
  JSON.stringify({
    report: existsSync(directory + "report.html"),
    handoff: existsSync(directory + "handoff.zip"),
  }),
);
