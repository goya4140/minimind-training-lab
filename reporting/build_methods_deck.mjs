// Internal methods-layout draft. Run with bundled Node and ARTIFACT_SKILL_DIR.
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const root = path.resolve(process.env.MINIMIND_REPORT_ROOT ?? path.join(path.dirname(fileURLToPath(import.meta.url)), ".."));
const buildDir = path.join(root, ".report-build");
const revision = process.argv[2];
if (!/^[a-z0-9-]+$/.test(revision ?? "")) throw new Error("Pass a new revision name, e.g. methods-v1");
const output = path.join(buildDir, revision);
await fs.mkdir(output, { recursive: false });
const contentPath = path.join(root, "reporting/methods-content.json");
const contentBytes = await fs.readFile(contentPath);
const content = JSON.parse(contentBytes);
if (content.schema !== 1 || content.status !== "methods-draft" || !content.missing_final_sections.length) {
  throw new Error("Methods-only builder cannot publish a final results report");
}
const FONT = "Arial Unicode MS";
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const nativeTables = [];

function text(slide, value, x, y, width, height, size = 28, color = "#263746", bold = false) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = { fontSize: size, typeface: FONT, color, bold, insets: 0, wrap: "word" };
  return shape;
}

const cover = deck.slides.add();
cover.background.fill = "#FFFFFF";
text(cover, content.title, 80, 180, 1120, 110, 58, "#172F42", true);
text(cover, content.subtitle, 80, 310, 1120, 65, 36);
text(cover, content.scope, 80, 420, 1060, 100, 25);
cover.speakerNotes.text = `Run ${content.run_id}\nTraining commit ${content.training_commit}\nSource reporting/methods-content.json\nMethods draft only; final results pending.`;

for (const [index, topic] of content.sections.entries()) {
  const slide = deck.slides.add();
  slide.background.fill = "#FFFFFF";
  text(slide, topic.title, 64, 40, 1152, 72, 44, "#172F42", true);
  const isImage = Boolean(topic.image);
  if (isImage) {
    const bytes = await fs.readFile(path.join(root, topic.image.path));
    if (createHash("sha256").update(bytes).digest("hex") !== topic.image.sha256) {
      throw new Error("Evidence screenshot checksum mismatch");
    }
    slide.images.add({
      blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      contentType: "image/png", alt: topic.image.caption, fit: "contain",
      position: { left: 60, top: 124, width: 830, height: 570 },
    });
    text(slide, topic.summary, 930, 160, 290, 360, 27);
  } else {
    text(slide, topic.summary, 64, 126, 1152, 106, 29);
    const data = topic.table;
    if (data) {
      const values = [data.headers, ...data.rows];
      const table = slide.tables.add({
        rows: values.length, columns: data.headers.length,
        left: 64, top: 260, width: 1152, height: 360,
        columnWidths: data.widths.map(w => w * 1152), values,
      });
      table.borders.assign({ fill: "#D9D9D9", width: 1, style: "solid" });
      for (let r = 0; r < values.length; r++) {
        table.rows[r].height = 360 / values.length;
        for (let c = 0; c < data.headers.length; c++) {
          const cell = table.getCell(r, c);
          cell.fill = r === 0 ? "#20394D" : (r % 2 ? "#FFFFFF" : "#F1F5F8");
          cell.text.style = { fontSize: 25, typeface: FONT, bold: r === 0, color: r === 0 ? "#FFFFFF" : "#172F42" };
        }
      }
      nativeTables.push(index + 2);
    }
  }
  // Full explanations and immutable source links remain in the speaker notes.
  slide.speakerNotes.text = [
    `Run ${content.run_id}`, `Training commit ${content.training_commit}`, topic.id,
    ...topic.paragraphs,
    ...topic.sources.map(source => `https://github.com/goya4140/minimind-training-lab/blob/${/^(src|scripts|configs)\//.test(source) ? content.training_commit : content.documentation_commit}/${source}`),
    "Methods-only draft. Six-stage final results and complete evaluation still required.",
  ].join("\n\n");
}

for (const [index, slide] of deck.slides.items.entries()) {
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(output, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
const candidatePath = path.join(output, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const skillDir = process.env.ARTIFACT_SKILL_DIR;
const runtimePython = process.env.ARTIFACT_RUNTIME_PYTHON;
if (!skillDir || !runtimePython) throw new Error("Pass bundled skill/runtime paths for finalization");
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
await fs.mkdir(path.join(output, "export"));
const finalPath = path.join(output, "export/methods-layout.pptx");
const finalized = await finalizePresentation({
  workspaceDir: root, candidatePath, finalPath, pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit",
    ...nativeTables.flatMap(number => ["--require-native-table-slide", String(number)])],
  explicitTotalSlideCount: content.sections.length + 1,
  requiredNativeTableOwnerSlides: nativeTables,
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(output, "validation.json"),
});
await fs.writeFile(path.join(output, "source-binding.json"), JSON.stringify({
  content_sha256: createHash("sha256").update(contentBytes).digest("hex"),
  run_id: content.run_id, training_commit: content.training_commit,
  status: content.status, pending: content.missing_final_sections,
}, null, 2));
console.log(JSON.stringify({ finalPath, status: content.status, slides: deck.slides.items.length, validated: Boolean(finalized) }));
