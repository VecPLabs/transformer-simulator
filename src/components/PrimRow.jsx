import { useState } from "react";
import { PRIM } from "../data/primitives";
import { DIM_OPTIONS, resolve } from "../data/dimensions";
import { fmt } from "../utils/format";
import { font } from "./styles";
import InfoTooltip from "./InfoTooltip";

function DimSelect({ value, globalCfg, onChange }) {
  const isCustomInt = typeof value === "number" && !DIM_OPTIONS.some(o => o.value === value);
  const [showCustom, setShowCustom] = useState(isCustomInt);
  const [customVal, setCustomVal] = useState(isCustomInt ? value : "");

  if (showCustom) {
    return (
      <div style={{ display: "flex", gap: 2, alignItems: "center" }}>
        <input
          type="number" autoFocus min={1} step={1}
          value={customVal}
          onChange={e => { const n = Number(e.target.value); setCustomVal(e.target.value); if (n > 0) onChange(n); }}
          onKeyDown={e => { if (e.key === "Escape") { setShowCustom(false); } }}
          style={{ width: 52, background: "#0c1220", border: "1px solid #22c55e33", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "1px 3px", borderRadius: 2 }}
        />
        <button onClick={() => setShowCustom(false)} style={{ background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 7, padding: 0 }}>↩</button>
      </div>
    );
  }

  return (
    <select value={typeof value === "number" ? "__custom__" : value} onChange={e => {
      const v = e.target.value;
      if (v === "__custom__") {
        setCustomVal(resolve(value, globalCfg));
        setShowCustom(true);
      } else {
        onChange(isNaN(Number(v)) ? v : Number(v));
      }
    }}
      style={{ background: "#0c1220", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "1px 2px", borderRadius: 2 }}>
      {DIM_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label} ({resolve(o.value, globalCfg)})</option>)}
      {isCustomInt && <option value="__custom__">Custom: {value}</option>}
      <option value="__custom__">{isCustomInt ? "" : "Custom..."}</option>
    </select>
  );
}

export default function PrimRow({ prim, globalCfg, onUpdate, onRemove, onMove, isFirst, isLast, dragH }) {
  const reg = PRIM[prim.type];
  if (!reg) return null;
  const p = reg.params(prim.cfg, globalCfg);
  const sh = reg.shape(prim.cfg, globalCfg);
  const [editing, setEditing] = useState(false);

  const updateCfg = (key, val) => {
    onUpdate({ ...prim, cfg: { ...prim.cfg, [key]: val } });
  };
  const updateLabel = (val) => onUpdate({ ...prim, label: val });

  return (
    <div
      draggable
      {...dragH}
      style={{
        display: "flex", flexDirection: "column",
        background: "#080c16", borderLeft: `2px solid ${reg.color}44`,
        borderRadius: 3, marginBottom: 2, cursor: "grab",
        transition: "border-color 0.12s",
      }}
      onMouseEnter={e => e.currentTarget.style.borderLeftColor = reg.color}
      onMouseLeave={e => e.currentTarget.style.borderLeftColor = `${reg.color}44`}
    >
      <div style={{ display: "flex", alignItems: "center", padding: "4px 6px", gap: 5 }}>
        <span style={{ fontSize: 11, color: reg.color, width: 16, textAlign: "center", flexShrink: 0 }}>{reg.icon}</span>
        {editing ? (
          <input value={prim.label} onChange={e => updateLabel(e.target.value)} onBlur={() => setEditing(false)} autoFocus
            style={{ flex: 1, background: "#0c1220", border: "1px solid #172035", color: "#c8d6e5", fontFamily: font, fontSize: 10, padding: "1px 4px", borderRadius: 2, minWidth: 0 }} />
        ) : (
          <span onClick={() => setEditing(true)} style={{ flex: 1, fontSize: 10, color: "#a0b4c8", cursor: "text", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", minWidth: 0 }} title={reg.desc}>
            {prim.label}
          </span>
        )}
        <span style={{ fontSize: 8, color: "#3e5775", fontStyle: "italic", whiteSpace: "nowrap", flexShrink: 0 }}>{sh}</span>
        {p > 0 && <span style={{ fontSize: 9, color: "#22c55e", fontWeight: 600, whiteSpace: "nowrap", flexShrink: 0 }}>{fmt(p)}</span>}
        <div style={{ display: "flex", gap: 0, flexShrink: 0 }}>
          <button onClick={e => { e.stopPropagation(); onMove(-1); }} disabled={isFirst}
            style={{ background: "none", border: "none", color: isFirst ? "#111827" : "#3e5775", cursor: isFirst ? "default" : "pointer", fontSize: 7, padding: "0 1px" }}>▲</button>
          <button onClick={e => { e.stopPropagation(); onMove(1); }} disabled={isLast}
            style={{ background: "none", border: "none", color: isLast ? "#111827" : "#3e5775", cursor: isLast ? "default" : "pointer", fontSize: 7, padding: "0 1px" }}>▼</button>
        </div>
        {reg.info && <InfoTooltip text={{ type: prim.type, info: reg.info, desc: reg.desc }} color={reg.color} />}
        {reg.configFields.length > 0 && (
          <button onClick={e => { e.stopPropagation(); onUpdate({ ...prim, _showCfg: !prim._showCfg }); }}
            style={{ background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 8, padding: "0 2px" }}>⚙</button>
        )}
        <button onClick={e => { e.stopPropagation(); onRemove(); }}
          style={{ background: "none", border: "none", color: "#3a2020", cursor: "pointer", fontSize: 10, padding: "0 2px", fontWeight: 700 }}>×</button>
      </div>

      {/* Config fields */}
      {prim._showCfg && reg.configFields.length > 0 && (
        <div style={{ padding: "2px 6px 4px 24px", display: "flex", flexWrap: "wrap", gap: 6 }}>
          {reg.configFields.map(f => (
            <div key={f.key} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ fontSize: 8, color: "#3e5775" }}>{f.label}:</span>
              {f.type === "dimSelect" ? (
                <DimSelect value={prim.cfg[f.key]} globalCfg={globalCfg} onChange={v => updateCfg(f.key, v)} />
              ) : f.type === "number" ? (
                <input type="number" value={prim.cfg[f.key]} min={f.min} max={f.max} step={f.step}
                  onChange={e => updateCfg(f.key, Number(e.target.value))}
                  style={{ width: 50, background: "#0c1220", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "1px 3px", borderRadius: 2 }} />
              ) : (
                <input type="text" value={prim.cfg[f.key]} onChange={e => updateCfg(f.key, e.target.value)}
                  style={{ width: 80, background: "#0c1220", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "1px 3px", borderRadius: 2 }} />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
