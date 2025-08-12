import fs from "fs";
export function loadJsonl(path) {
  const lines = fs.readFileSync(path, "utf8")
    .split("\n")
    .map(l => l.trim())
    .filter(Boolean);
  return lines.map(l => JSON.parse(l));
}