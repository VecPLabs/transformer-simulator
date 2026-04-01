import { useState } from "react";
import { PRIM } from "../data/primitives";
import { BLOCK_TEMPLATES } from "../data/blockTemplates";
import { fmt } from "../utils/format";
import { uid } from "../utils/uid";
import { S, font } from "./styles";
import BlockCard from "./BlockCard";

function makePrim(type, cfgOverride, label) {
  const reg = PRIM[type];
  return {
    id: uid(), type, label: label || reg.name,
    cfg: { ...reg.defaultCfg, ...(cfgOverride || {}) },
  };
}

function makeBlock(templateKey) {
  const t = BLOCK_TEMPLATES[templateKey];
  return {
    id: uid(), name: t.name, color: t.color, collapsed: false,
    primitives: t.primitives.map(p => makePrim(p.type, p.cfg, p.label)),
  };
}

export default function LayerCard({ layer, idx, globalCfg, total, onUpdate, onRemove, onDuplicate, onMoveLayer }) {
  const [addingBlock, setAddingBlock] = useState(false);

  const layerParams = layer.blocks.reduce((s, b) =>
    s + b.primitives.reduce((bs, p) => bs + (PRIM[p.type]?.params(p.cfg, globalCfg) || 0), 0), 0);

  const updateBlock = (bIdx, block) => {
    const blocks = [...layer.blocks]; blocks[bIdx] = block;
    onUpdate({ ...layer, blocks });
  };
  const removeBlock = (bIdx) => onUpdate({ ...layer, blocks: layer.blocks.filter((_, i) => i !== bIdx) });
  const duplicateBlock = (bIdx) => {
    const src = layer.blocks[bIdx];
    const dup = { ...src, id: uid(), primitives: src.primitives.map(p => ({ ...p, id: uid() })) };
    onUpdate({ ...layer, blocks: [...layer.blocks.slice(0, bIdx + 1), dup, ...layer.blocks.slice(bIdx + 1)] });
  };
  const moveBlock = (bIdx, dir) => {
    const to = bIdx + dir; if (to < 0 || to >= layer.blocks.length) return;
    const blocks = [...layer.blocks]; [blocks[bIdx], blocks[to]] = [blocks[to], blocks[bIdx]];
    onUpdate({ ...layer, blocks });
  };
  const addBlock = (templateKey) => {
    onUpdate({ ...layer, blocks: [...layer.blocks, makeBlock(templateKey)] });
    setAddingBlock(false);
  };

  return (
    <div style={{ ...S.panel, marginBottom: 6, padding: 0, borderColor: "#172035" }}>
      <div style={{
        display: "flex", alignItems: "center", padding: "5px 8px", gap: 6,
        background: "#0a0f1c", borderBottom: layer.collapsed ? "none" : "1px solid #172035",
        borderRadius: layer.collapsed ? 6 : "6px 6px 0 0",
      }}>
        <button onClick={() => onUpdate({ ...layer, collapsed: !layer.collapsed })}
          style={{ background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 9, padding: 0 }}>
          {layer.collapsed ? "▸" : "▾"}
        </button>
        <div style={{
          width: 20, height: 20, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center",
          background: "linear-gradient(135deg, #22c55e0c, #06b6d40c)", border: "1px solid #172035",
          fontSize: 9, fontWeight: 700, color: "#22c55e",
        }}>{idx + 1}</div>
        <span style={{ fontSize: 10, color: "#8899aa", flex: 1 }}>
          Layer {idx + 1}
          <span style={{ color: "#2a3f55", marginLeft: 6, fontSize: 8 }}>{layer.blocks.length} blocks</span>
        </span>
        {layer.blocks.length > 1 && (
          <select
            value={layer.topology || "sequential"}
            onChange={e => onUpdate({ ...layer, topology: e.target.value, parallelCount: layer.parallelCount || Math.max(1, layer.blocks.length - 1) })}
            onClick={e => e.stopPropagation()}
            title="Block topology for code export"
            style={{ background: "#080c16", border: "1px solid #172035", color: "#6b8299", fontFamily: font, fontSize: 7, padding: "1px 2px", borderRadius: 2 }}
          >
            <option value="sequential">Sequential</option>
            <option value="parallel">Parallel</option>
            <option value="parallel_then_sequential">Parallel → Seq</option>
          </select>
        )}
        {layer.topology === "parallel_then_sequential" && (
          <input
            type="number" min={1} max={layer.blocks.length}
            value={layer.parallelCount || Math.max(1, layer.blocks.length - 1)}
            onChange={e => onUpdate({ ...layer, parallelCount: Math.max(1, Math.min(layer.blocks.length, Number(e.target.value))) })}
            onClick={e => e.stopPropagation()}
            title="Number of parallel blocks"
            style={{ width: 24, background: "#080c16", border: "1px solid #172035", color: "#6b8299", fontFamily: font, fontSize: 7, padding: "1px 2px", borderRadius: 2, textAlign: "center" }}
          />
        )}
        <span style={{ fontSize: 9, color: "#22c55e", fontWeight: 600 }}>{fmt(layerParams)}</span>
        <div style={{ display: "flex", gap: 1 }}>
          <button onClick={() => onMoveLayer(-1)} disabled={idx === 0}
            style={{ background: "none", border: "none", color: idx === 0 ? "#111827" : "#3e5775", cursor: idx === 0 ? "default" : "pointer", fontSize: 7, padding: "0 1px" }}>▲</button>
          <button onClick={() => onMoveLayer(1)} disabled={idx === total - 1}
            style={{ background: "none", border: "none", color: idx === total - 1 ? "#111827" : "#3e5775", cursor: idx === total - 1 ? "default" : "pointer", fontSize: 7, padding: "0 1px" }}>▼</button>
          <button onClick={onDuplicate} style={{ background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 9, padding: "0 2px" }}>⧉</button>
          <button onClick={onRemove} style={{ background: "none", border: "none", color: "#3a2020", cursor: "pointer", fontSize: 11, padding: "0 2px", fontWeight: 700 }}>×</button>
        </div>
      </div>

      {!layer.collapsed && (
        <div style={{ padding: "6px 6px" }}>
          {layer.blocks.map((block, bIdx) => (
            <BlockCard
              key={block.id} block={block} globalCfg={globalCfg}
              onUpdate={b => updateBlock(bIdx, b)}
              onRemove={() => removeBlock(bIdx)}
              onDuplicate={() => duplicateBlock(bIdx)}
              onMove={dir => moveBlock(bIdx, dir)}
              isFirst={bIdx === 0} isLast={bIdx === layer.blocks.length - 1}
            />
          ))}

          {addingBlock ? (
            <div style={{ ...S.panel, padding: 8, background: "#080c16", border: "1px solid #22c55e22", marginTop: 4 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ ...S.label, color: "#22c55e" }}>Add Block</span>
                <button onClick={() => setAddingBlock(false)} style={{ background: "none", border: "none", color: "#3a2020", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>×</button>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                {Object.entries(BLOCK_TEMPLATES).map(([k, t]) => (
                  <button key={k} onClick={() => addBlock(k)} style={{
                    background: `${t.color}0a`, border: `1px solid ${t.color}25`, color: t.color,
                    borderRadius: 3, padding: "3px 7px", fontSize: 9, cursor: "pointer", fontFamily: font,
                  }}
                    onMouseEnter={e => e.currentTarget.style.background = `${t.color}18`}
                    onMouseLeave={e => e.currentTarget.style.background = `${t.color}0a`}
                  >{t.name}</button>
                ))}
              </div>
            </div>
          ) : (
            <button onClick={() => setAddingBlock(true)} style={{
              width: "100%", background: "none", border: "1px dashed #172035", color: "#2a3f55",
              borderRadius: 3, padding: "4px 0", fontSize: 8, cursor: "pointer", fontFamily: font, marginTop: 2,
            }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "#22c55e33"; e.currentTarget.style.color = "#22c55e"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "#172035"; e.currentTarget.style.color = "#2a3f55"; }}
            >+ Add Block</button>
          )}
        </div>
      )}
    </div>
  );
}
