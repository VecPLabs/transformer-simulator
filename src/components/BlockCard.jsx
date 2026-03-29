import { useState } from "react";
import { PRIM } from "../data/primitives";
import { BLOCK_TEMPLATES } from "../data/blockTemplates";
import { fmt } from "../utils/format";
import { uid } from "../utils/uid";
import { S, font } from "./styles";
import PrimRow from "./PrimRow";
import PrimPalette from "./PrimPalette";

function makePrim(type, cfgOverride, label) {
  const reg = PRIM[type];
  return {
    id: uid(), type, label: label || reg.name,
    cfg: { ...reg.defaultCfg, ...(cfgOverride || {}) },
  };
}

export default function BlockCard({ block, globalCfg, onUpdate, onRemove, onDuplicate, onMove, isFirst, isLast }) {
  const [addingPrim, setAddingPrim] = useState(false);
  const [dragOver, setDragOver] = useState(null);

  const blockParams = block.primitives.reduce((s, p) => {
    const reg = PRIM[p.type];
    return s + (reg ? reg.params(p.cfg, globalCfg) : 0);
  }, 0);

  const updatePrim = (idx, prim) => {
    const prims = [...block.primitives];
    prims[idx] = prim;
    onUpdate({ ...block, primitives: prims });
  };
  const removePrim = (idx) => onUpdate({ ...block, primitives: block.primitives.filter((_, i) => i !== idx) });
  const movePrim = (idx, dir) => {
    const to = idx + dir;
    if (to < 0 || to >= block.primitives.length) return;
    const prims = [...block.primitives];
    [prims[idx], prims[to]] = [prims[to], prims[idx]];
    onUpdate({ ...block, primitives: prims });
  };
  const addPrim = (type) => {
    onUpdate({ ...block, primitives: [...block.primitives, makePrim(type)] });
    setAddingPrim(false);
  };

  const handleDragStart = (e, idx) => {
    e.dataTransfer.setData("text/plain", JSON.stringify({ blockId: block.id, primIdx: idx }));
  };
  const handleDragOver = (e, idx) => { e.preventDefault(); setDragOver(idx); };
  const handleDrop = (e, toIdx) => {
    e.preventDefault(); setDragOver(null);
    try {
      const data = JSON.parse(e.dataTransfer.getData("text/plain"));
      if (data.blockId === block.id) {
        const prims = [...block.primitives];
        const [moved] = prims.splice(data.primIdx, 1);
        prims.splice(toIdx > data.primIdx ? toIdx - 1 : toIdx, 0, moved);
        onUpdate({ ...block, primitives: prims });
      }
    } catch {}
  };

  const [editingName, setEditingName] = useState(false);

  return (
    <div style={{
      ...S.panel, marginBottom: 4, borderLeft: `3px solid ${block.color}`,
      boxShadow: `0 0 8px ${block.color}08`,
    }}>
      {/* Block header */}
      <div style={{
        display: "flex", alignItems: "center", padding: "5px 8px", gap: 6,
        background: "#0a0f1c", borderRadius: block.collapsed ? "4px" : "4px 4px 0 0",
        borderBottom: block.collapsed ? "none" : "1px solid #172035",
      }}>
        <button onClick={() => onUpdate({ ...block, collapsed: !block.collapsed })}
          style={{ background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 9, padding: 0 }}>
          {block.collapsed ? "▸" : "▾"}
        </button>
        {editingName ? (
          <input value={block.name} onChange={e => onUpdate({ ...block, name: e.target.value })} onBlur={() => setEditingName(false)} autoFocus
            style={{ flex: 1, background: "#0c1220", border: "1px solid #172035", color: "#c8d6e5", fontFamily: font, fontSize: 10, padding: "1px 4px", borderRadius: 2 }} />
        ) : (
          <span onDoubleClick={() => setEditingName(true)} style={{ flex: 1, fontSize: 10, fontWeight: 600, color: block.color, cursor: "default" }} title="Double-click to rename">
            {block.name}
            <span style={{ color: "#2a3f55", fontWeight: 400, marginLeft: 6, fontSize: 9 }}>{block.primitives.length} ops</span>
          </span>
        )}
        <span style={{ fontSize: 9, color: blockParams > 0 ? "#22c55e" : "#1a2744", fontWeight: 600 }}>
          {blockParams > 0 ? fmt(blockParams) : "0"}
        </span>
        <div style={{ display: "flex", gap: 1, flexShrink: 0 }}>
          <button onClick={() => onMove(-1)} disabled={isFirst}
            style={{ background: "none", border: "none", color: isFirst ? "#111827" : "#3e5775", cursor: isFirst ? "default" : "pointer", fontSize: 7, padding: "0 1px" }}>▲</button>
          <button onClick={() => onMove(1)} disabled={isLast}
            style={{ background: "none", border: "none", color: isLast ? "#111827" : "#3e5775", cursor: isLast ? "default" : "pointer", fontSize: 7, padding: "0 1px" }}>▼</button>
          <button onClick={onDuplicate} title="Duplicate block"
            style={{ background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 9, padding: "0 2px" }}>⧉</button>
          <button onClick={onRemove}
            style={{ background: "none", border: "none", color: "#3a2020", cursor: "pointer", fontSize: 11, padding: "0 2px", fontWeight: 700 }}>×</button>
        </div>
      </div>

      {/* Primitives */}
      {!block.collapsed && (
        <div style={{ padding: "4px 6px" }}>
          {block.primitives.map((prim, idx) => (
            <div key={prim.id}
              onDragOver={e => handleDragOver(e, idx)}
              onDrop={e => handleDrop(e, idx)}
              onDragLeave={() => setDragOver(null)}
            >
              {dragOver === idx && <div style={{ height: 2, background: "#22c55e", borderRadius: 1, margin: "1px 0" }} />}
              <PrimRow
                prim={prim} globalCfg={globalCfg}
                onUpdate={p => updatePrim(idx, p)}
                onRemove={() => removePrim(idx)}
                onMove={dir => movePrim(idx, dir)}
                isFirst={idx === 0}
                isLast={idx === block.primitives.length - 1}
                dragH={{
                  onDragStart: e => handleDragStart(e, idx),
                  onDragEnd: () => setDragOver(null),
                }}
              />
            </div>
          ))}
          <div onDragOver={e => handleDragOver(e, block.primitives.length)} onDrop={e => handleDrop(e, block.primitives.length)} onDragLeave={() => setDragOver(null)} style={{ minHeight: 4 }}>
            {dragOver === block.primitives.length && <div style={{ height: 2, background: "#22c55e", borderRadius: 1 }} />}
          </div>

          {addingPrim ? (
            <PrimPalette onAdd={addPrim} onClose={() => setAddingPrim(false)} />
          ) : (
            <div style={{ display: "flex", gap: 4, marginTop: 2 }}>
              <button onClick={() => setAddingPrim(true)} style={{
                flex: 1, background: "none", border: "1px dashed #172035", color: "#2a3f55",
                borderRadius: 3, padding: "3px 0", fontSize: 8, cursor: "pointer", fontFamily: font,
              }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = "#22c55e33"; e.currentTarget.style.color = "#22c55e"; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = "#172035"; e.currentTarget.style.color = "#2a3f55"; }}
              >+ Add Primitive</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
