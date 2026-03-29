import { S, font } from "./styles";

const HELP_ITEMS = [
  {
    title: "Primitives",
    desc: "The atomic building blocks — linear projections, activations, norms, and operations. Each has configurable dimensions and shows its parameter count.",
    icon: "─",
  },
  {
    title: "Blocks",
    desc: "Groups of primitives that form a logical unit (e.g., 'Pre-Norm + GQA + Residual'). Add from templates or build custom. Double-click names to rename.",
    icon: "▣",
  },
  {
    title: "Layers",
    desc: "Repeated units of blocks. Most transformers stack identical layers. Use batch ops to apply changes across ranges.",
    icon: "▤",
  },
  {
    title: "Drag & Reorder",
    desc: "Drag primitives within a block to reorder. Use arrow buttons to move layers, blocks, or primitives up/down.",
    icon: "↕",
  },
  {
    title: "Dimensions",
    desc: "Primitives reference named dimensions (Hidden Dim, FFN Dim, etc.) so configs stay valid when you change global settings. Use 'Custom...' for arbitrary values.",
    icon: "⚙",
  },
  {
    title: "Shape Tracking",
    desc: "Tensor shapes are traced through each block. Red warnings appear when a primitive's expected input doesn't match the previous output.",
    icon: "⚠",
  },
  {
    title: "Share & Import",
    desc: "Export configs as JSON files, clipboard text, or shareable URLs. Import from any of these. Architecture metadata (name, author, notes) travels with the config.",
    icon: "⌗",
  },
  {
    title: "Shortcuts",
    desc: "Ctrl+Z / Ctrl+Y: undo/redo. Ctrl+S: share. Ctrl+Shift+E/C: expand/collapse all layers.",
    icon: "⌨",
  },
];

export default function HelpOverlay({ onClose }) {
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(0,0,0,0.75)", backdropFilter: "blur(4px)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 16,
    }} onClick={onClose}>
      <div style={{
        ...S.panel, width: "100%", maxWidth: 400, maxHeight: "85vh", overflowY: "auto",
        padding: 0, background: "var(--header-bg)", border: "1px solid var(--accent-green, #22c55e)33",
        boxShadow: "0 0 40px rgba(34,197,94,0.08)",
      }} onClick={e => e.stopPropagation()}>
        <div style={{
          padding: "12px 14px", borderBottom: "1px solid var(--border)",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <div style={{ fontSize: 7, color: "var(--accent-green, #22c55e)", letterSpacing: "0.3em", textTransform: "uppercase", fontWeight: 600 }}>Getting Started</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "var(--text-bright, #e2eaf2)", marginTop: 2 }}>How It Works</div>
          </div>
          <button onClick={onClose} style={{
            background: "none", border: "none", color: "var(--muted)", cursor: "pointer", fontSize: 16, fontWeight: 700, padding: "0 4px",
          }}>×</button>
        </div>

        <div style={{ padding: "8px 14px 14px" }}>
          {HELP_ITEMS.map((item, i) => (
            <div key={i} style={{
              display: "flex", gap: 10, padding: "8px 0",
              borderBottom: i < HELP_ITEMS.length - 1 ? "1px solid var(--border)" : "none",
            }}>
              <div style={{
                width: 24, height: 24, borderRadius: 4, flexShrink: 0,
                display: "flex", alignItems: "center", justifyContent: "center",
                background: "var(--panel-bg-deep)", border: "1px solid var(--border)",
                fontSize: 11, color: "var(--accent-green, #22c55e)",
              }}>{item.icon}</div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text)", marginBottom: 2 }}>{item.title}</div>
                <div style={{ fontSize: 8, color: "var(--muted)", lineHeight: 1.5 }}>{item.desc}</div>
              </div>
            </div>
          ))}
        </div>

        <div style={{
          padding: "8px 14px", borderTop: "1px solid var(--border)",
          textAlign: "center", fontSize: 8, color: "var(--muted-deep)",
        }}>
          Press ? or click the help button anytime to see this again
        </div>
      </div>
    </div>
  );
}
