import { useState } from "react";
import { PRIM } from "../data/primitives";
import { BLOCK_TEMPLATES } from "../data/blockTemplates";
import { uid } from "../utils/uid";
import { S, font } from "./styles";

function makePrim(type, cfgOverride, label) {
  const reg = PRIM[type];
  return { id: uid(), type, label: label || reg.name, cfg: { ...reg.defaultCfg, ...(cfgOverride || {}) } };
}

function makeBlock(templateKey) {
  const t = BLOCK_TEMPLATES[templateKey];
  return { id: uid(), name: t.name, color: t.color, collapsed: false, primitives: t.primitives.map(p => makePrim(p.type, p.cfg, p.label)) };
}

export default function BatchOps({ layers, onUpdateLayers, showToast }) {
  const [op, setOp] = useState("template"); // "template" | "replace" | "clone"
  const [rangeStart, setRangeStart] = useState(1);
  const [rangeEnd, setRangeEnd] = useState(layers.length);
  const [selectedTemplate, setSelectedTemplate] = useState(Object.keys(BLOCK_TEMPLATES)[0]);
  const [replaceFrom, setReplaceFrom] = useState("rmsnorm");
  const [replaceTo, setReplaceTo] = useState("layernorm");
  const [sourceLayer, setSourceLayer] = useState(1);

  const clampRange = () => {
    const s = Math.max(1, Math.min(rangeStart, layers.length));
    const e = Math.max(s, Math.min(rangeEnd, layers.length));
    return [s - 1, e]; // convert to 0-indexed start, exclusive end
  };

  const applyTemplate = () => {
    const [start, end] = clampRange();
    const newLayers = layers.map((l, i) => {
      if (i < start || i >= end) return l;
      return { ...l, blocks: [...l.blocks, makeBlock(selectedTemplate)] };
    });
    onUpdateLayers(newLayers);
    showToast(`Added "${BLOCK_TEMPLATES[selectedTemplate].name}" to layers ${start + 1}-${end}`);
  };

  const applyReplace = () => {
    const [start, end] = clampRange();
    if (!PRIM[replaceFrom] || !PRIM[replaceTo]) return;
    let count = 0;
    const newLayers = layers.map((l, i) => {
      if (i < start || i >= end) return l;
      return {
        ...l,
        blocks: l.blocks.map(b => ({
          ...b,
          primitives: b.primitives.map(p => {
            if (p.type === replaceFrom) {
              count++;
              const newReg = PRIM[replaceTo];
              return { ...p, id: uid(), type: replaceTo, label: newReg.name, cfg: { ...newReg.defaultCfg } };
            }
            return p;
          }),
        })),
      };
    });
    onUpdateLayers(newLayers);
    showToast(`Replaced ${count} ${PRIM[replaceFrom].name} → ${PRIM[replaceTo].name}`);
  };

  const applyClone = () => {
    const [start, end] = clampRange();
    const srcIdx = Math.max(0, Math.min(sourceLayer - 1, layers.length - 1));
    const src = layers[srcIdx];
    if (!src) return;
    const newLayers = layers.map((l, i) => {
      if (i < start || i >= end || i === srcIdx) return l;
      return {
        ...l,
        blocks: src.blocks.map(b => ({
          ...b, id: uid(),
          primitives: b.primitives.map(p => ({ ...p, id: uid() })),
        })),
      };
    });
    onUpdateLayers(newLayers);
    showToast(`Cloned layer ${sourceLayer} structure to layers ${start + 1}-${end}`);
  };

  const btnStyle = (active) => ({
    background: active ? "#172035" : "none", border: "1px solid #172035",
    color: active ? "#c8d6e5" : "#3e5775", borderRadius: 3,
    padding: "3px 8px", fontSize: 8, fontFamily: font, cursor: "pointer",
    fontWeight: active ? 600 : 400,
  });

  const inputStyle = {
    width: 36, background: "#080c16", border: "1px solid #172035", color: "#a0b4c8",
    fontFamily: font, fontSize: 9, padding: "2px 4px", borderRadius: 2, textAlign: "center",
  };

  return (
    <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
      <div style={{ ...S.label, marginBottom: 6, color: "#f97316" }}>Batch Operations</div>

      {/* Op selector */}
      <div style={{ display: "flex", gap: 3, marginBottom: 8 }}>
        <button onClick={() => setOp("template")} style={btnStyle(op === "template")}>Add Block</button>
        <button onClick={() => setOp("replace")} style={btnStyle(op === "replace")}>Replace Prim</button>
        <button onClick={() => setOp("clone")} style={btnStyle(op === "clone")}>Clone Layer</button>
      </div>

      {/* Range selector */}
      <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 8 }}>
        <span style={{ fontSize: 8, color: "#3e5775" }}>Layers</span>
        <input type="number" min={1} max={layers.length} value={rangeStart} onChange={e => setRangeStart(Number(e.target.value))} style={inputStyle} />
        <span style={{ fontSize: 8, color: "#3e5775" }}>to</span>
        <input type="number" min={1} max={layers.length} value={rangeEnd} onChange={e => setRangeEnd(Number(e.target.value))} style={inputStyle} />
        <span style={{ fontSize: 7, color: "#2a3f55" }}>of {layers.length}</span>
      </div>

      {/* Op-specific UI */}
      {op === "template" && (
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <select value={selectedTemplate} onChange={e => setSelectedTemplate(e.target.value)}
            style={{ flex: 1, background: "#080c16", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "3px 4px", borderRadius: 2 }}>
            {Object.entries(BLOCK_TEMPLATES).map(([k, t]) => (
              <option key={k} value={k}>{t.name}</option>
            ))}
          </select>
          <button onClick={applyTemplate} style={{
            background: "linear-gradient(135deg, #f97316, #ea580c)", color: "#000",
            border: "none", borderRadius: 3, padding: "4px 10px", fontSize: 8,
            fontFamily: font, fontWeight: 700, cursor: "pointer",
          }}>Apply</button>
        </div>
      )}

      {op === "replace" && (
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <select value={replaceFrom} onChange={e => setReplaceFrom(e.target.value)}
            style={{ flex: 1, background: "#080c16", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "3px 4px", borderRadius: 2 }}>
            {Object.entries(PRIM).map(([k, r]) => (
              <option key={k} value={k}>{r.icon} {r.name}</option>
            ))}
          </select>
          <span style={{ fontSize: 8, color: "#3e5775" }}>→</span>
          <select value={replaceTo} onChange={e => setReplaceTo(e.target.value)}
            style={{ flex: 1, background: "#080c16", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "3px 4px", borderRadius: 2 }}>
            {Object.entries(PRIM).map(([k, r]) => (
              <option key={k} value={k}>{r.icon} {r.name}</option>
            ))}
          </select>
          <button onClick={applyReplace} style={{
            background: "linear-gradient(135deg, #f97316, #ea580c)", color: "#000",
            border: "none", borderRadius: 3, padding: "4px 10px", fontSize: 8,
            fontFamily: font, fontWeight: 700, cursor: "pointer",
          }}>Apply</button>
        </div>
      )}

      {op === "clone" && (
        <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
          <span style={{ fontSize: 8, color: "#3e5775" }}>Copy structure from layer</span>
          <input type="number" min={1} max={layers.length} value={sourceLayer} onChange={e => setSourceLayer(Number(e.target.value))} style={inputStyle} />
          <button onClick={applyClone} style={{
            background: "linear-gradient(135deg, #f97316, #ea580c)", color: "#000",
            border: "none", borderRadius: 3, padding: "4px 10px", fontSize: 8,
            fontFamily: font, fontWeight: 700, cursor: "pointer",
          }}>Apply</button>
        </div>
      )}
    </div>
  );
}
