import { PRIM, PRIM_CATEGORIES } from "../data/primitives";
import { S, font } from "./styles";

export default function PrimPalette({ onAdd, onClose }) {
  return (
    <div style={{ ...S.panel, padding: 8, marginTop: 4, border: "1px solid #22c55e22", background: "#080c16" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ ...S.label, color: "#22c55e" }}>Add Primitive</span>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#3a2020", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>×</button>
      </div>
      {PRIM_CATEGORIES.map(cat => {
        const items = Object.entries(PRIM).filter(([, r]) => r.category === cat.key);
        if (!items.length) return null;
        return (
          <div key={cat.key} style={{ marginBottom: 5 }}>
            <div style={{ ...S.label, fontSize: 8, marginBottom: 2, color: "#2a3f55" }}>{cat.label}</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
              {items.map(([k, r]) => (
                <button key={k} onClick={() => onAdd(k)} style={{
                  background: `${r.color}0a`, border: `1px solid ${r.color}25`,
                  color: r.color, borderRadius: 3, padding: "2px 6px",
                  fontSize: 9, cursor: "pointer", fontFamily: font,
                  display: "flex", alignItems: "center", gap: 3,
                }}
                  onMouseEnter={e => e.currentTarget.style.background = `${r.color}18`}
                  onMouseLeave={e => e.currentTarget.style.background = `${r.color}0a`}
                  title={r.desc}
                >
                  <span>{r.icon}</span>{r.name}
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
