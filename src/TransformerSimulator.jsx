import { useState, useMemo, useCallback, useRef, useEffect } from "react";

// ─── Primitive Registry (the atoms) ───────────────────────────────────
const PRIM = {
  linear: {
    name: "Linear Projection", icon: "─", color: "#ec4899", category: "projection",
    desc: "Wx (learnable matrix multiply)",
    defaultCfg: { inDim: "hiddenDim", outDim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.inDim, g) * resolve(cfg.outDim, g),
    shape: (cfg, g) => `[${resolve(cfg.inDim, g)}, ${resolve(cfg.outDim, g)}]`,
    configFields: [
      { key: "inDim", label: "Input Dim", type: "dimSelect" },
      { key: "outDim", label: "Output Dim", type: "dimSelect" },
    ],
  },
  linear_bias: {
    name: "Linear + Bias", icon: "━", color: "#f472b6", category: "projection",
    desc: "Wx + b (projection with bias)",
    defaultCfg: { inDim: "hiddenDim", outDim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.inDim, g) * resolve(cfg.outDim, g) + resolve(cfg.outDim, g),
    shape: (cfg, g) => `[${resolve(cfg.inDim, g)}, ${resolve(cfg.outDim, g)}] + b`,
    configFields: [
      { key: "inDim", label: "Input Dim", type: "dimSelect" },
      { key: "outDim", label: "Output Dim", type: "dimSelect" },
    ],
  },
  silu: {
    name: "SiLU / Swish", icon: "∿", color: "#facc15", category: "activation",
    desc: "x · σ(x)",
    defaultCfg: {}, params: () => 0, shape: () => "SiLU(x)",
    configFields: [],
  },
  gelu: {
    name: "GELU", icon: "≈", color: "#fbbf24", category: "activation",
    desc: "Gaussian Error Linear Unit",
    defaultCfg: {}, params: () => 0, shape: () => "GELU(x)",
    configFields: [],
  },
  relu: {
    name: "ReLU", icon: "∠", color: "#f59e0b", category: "activation",
    desc: "max(0, x)",
    defaultCfg: {}, params: () => 0, shape: () => "ReLU(x)",
    configFields: [],
  },
  softmax: {
    name: "Softmax", icon: "σ", color: "#fde68a", category: "activation",
    desc: "Normalized exponential",
    defaultCfg: {}, params: () => 0, shape: () => "softmax(x)",
    configFields: [],
  },
  gate_mul: {
    name: "Gate Multiply", icon: "⊙", color: "#a78bfa", category: "operation",
    desc: "Element-wise a ⊙ b (gating)",
    defaultCfg: {}, params: () => 0, shape: () => "a ⊙ b",
    configFields: [],
  },
  residual_add: {
    name: "Residual Add", icon: "⊞", color: "#6366f1", category: "operation",
    desc: "x + sublayer(x)",
    defaultCfg: {}, params: () => 0, shape: () => "x + f(x)",
    configFields: [],
  },
  scale_dot: {
    name: "Scaled Dot-Product", icon: "⊛", color: "#22d3ee", category: "operation",
    desc: "softmax(QKᵀ/√d)V",
    defaultCfg: { numHeads: "heads", headDim: "headDim" },
    params: () => 0,
    shape: (cfg, g) => `${resolve(cfg.numHeads, g)}h × ${resolve(cfg.headDim, g)}d`,
    configFields: [
      { key: "numHeads", label: "Heads", type: "dimSelect" },
      { key: "headDim", label: "Head Dim", type: "dimSelect" },
    ],
  },
  rmsnorm: {
    name: "RMSNorm", icon: "◇", color: "#22c55e", category: "norm",
    desc: "Root Mean Square normalization",
    defaultCfg: { dim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.dim, g),
    shape: (cfg, g) => `γ [${resolve(cfg.dim, g)}]`,
    configFields: [{ key: "dim", label: "Dim", type: "dimSelect" }],
  },
  layernorm: {
    name: "LayerNorm", icon: "◆", color: "#10b981", category: "norm",
    desc: "Layer norm with γ and β",
    defaultCfg: { dim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.dim, g) * 2,
    shape: (cfg, g) => `γ,β [${resolve(cfg.dim, g)}]`,
    configFields: [{ key: "dim", label: "Dim", type: "dimSelect" }],
  },
  dropout: {
    name: "Dropout", icon: "◌", color: "#64748b", category: "regularization",
    desc: "Random zeroing during training",
    defaultCfg: { rate: 0.1 },
    params: () => 0, shape: (cfg) => `p=${cfg.rate}`,
    configFields: [{ key: "rate", label: "Rate", type: "number", min: 0, max: 1, step: 0.01 }],
  },
  rope: {
    name: "RoPE", icon: "⟳", color: "#0ea5e9", category: "operation",
    desc: "Rotary Position Embedding (no params)",
    defaultCfg: {}, params: () => 0, shape: () => "rotate(x, θ)",
    configFields: [],
  },
  concat: {
    name: "Concat", icon: "⊕", color: "#8b5cf6", category: "operation",
    desc: "Concatenate along dim",
    defaultCfg: {}, params: () => 0, shape: () => "cat(a, b)",
    configFields: [],
  },
  split: {
    name: "Split / Branch", icon: "⑂", color: "#7c3aed", category: "operation",
    desc: "Fork tensor into parallel paths",
    defaultCfg: {}, params: () => 0, shape: () => "split(x)",
    configFields: [],
  },
  router: {
    name: "Router", icon: "⬡", color: "#f59e0b", category: "operation",
    desc: "Top-k expert routing",
    defaultCfg: { numExperts: "moeExperts", topK: "moeTopK", dim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.dim, g) * resolve(cfg.numExperts, g),
    shape: (cfg, g) => `top-${resolve(cfg.topK, g)} of ${resolve(cfg.numExperts, g)}`,
    configFields: [
      { key: "numExperts", label: "Experts", type: "dimSelect" },
      { key: "topK", label: "Top-K", type: "dimSelect" },
    ],
  },
  custom_note: {
    name: "Annotation", icon: "✎", color: "#475569", category: "meta",
    desc: "Non-functional note / label",
    defaultCfg: { text: "custom step" },
    params: () => 0, shape: (cfg) => cfg.text,
    configFields: [{ key: "text", label: "Note", type: "text" }],
  },
};

// Named dimension references
const DIM_OPTIONS = [
  { value: "hiddenDim", label: "Hidden Dim" },
  { value: "ffnDim", label: "FFN Dim" },
  { value: "heads", label: "Num Heads" },
  { value: "kvHeads", label: "KV Heads" },
  { value: "headDim", label: "Head Dim" },
  { value: "vocabSize", label: "Vocab Size" },
  { value: "moeExperts", label: "MoE Experts" },
  { value: "moeTopK", label: "MoE Top-K" },
];

function resolve(ref, globalCfg) {
  if (typeof ref === "number") return ref;
  if (ref === "headDim") return Math.floor(globalCfg.hiddenDim / globalCfg.heads);
  if (ref === "kvDim") return globalCfg.kvHeads * Math.floor(globalCfg.hiddenDim / globalCfg.heads);
  return globalCfg[ref] || 0;
}

const PRIM_CATEGORIES = [
  { key: "projection", label: "Projections" },
  { key: "activation", label: "Activations" },
  { key: "norm", label: "Norms" },
  { key: "operation", label: "Operations" },
  { key: "regularization", label: "Regularization" },
  { key: "meta", label: "Meta" },
];

// ─── Block Templates ──────────────────────────────────────────────────
const BLOCK_TEMPLATES = {
  mha: {
    name: "Multi-Head Attention", color: "#06b6d4",
    primitives: [
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_Q" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_K" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_V" },
      { type: "rope", cfg: {}, label: "RoPE" },
      { type: "scale_dot", cfg: { numHeads: "heads", headDim: "headDim" }, label: "Attention" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_O" },
    ],
  },
  gqa: {
    name: "Grouped Query Attention", color: "#0891b2",
    primitives: [
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_Q" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "kvDim" }, label: "W_K (GQA)" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "kvDim" }, label: "W_V (GQA)" },
      { type: "rope", cfg: {}, label: "RoPE" },
      { type: "scale_dot", cfg: { numHeads: "heads", headDim: "headDim" }, label: "GQA Attn" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_O" },
    ],
  },
  ffn_swiglu: {
    name: "FFN (SwiGLU)", color: "#a855f7",
    primitives: [
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "ffnDim" }, label: "W_gate" },
      { type: "silu", cfg: {}, label: "SiLU" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "ffnDim" }, label: "W_up" },
      { type: "gate_mul", cfg: {}, label: "Gate ⊙ Up" },
      { type: "linear", cfg: { inDim: "ffnDim", outDim: "hiddenDim" }, label: "W_down" },
    ],
  },
  ffn_gelu: {
    name: "FFN (GELU)", color: "#8b5cf6",
    primitives: [
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "ffnDim" }, label: "W_up" },
      { type: "gelu", cfg: {}, label: "GELU" },
      { type: "linear", cfg: { inDim: "ffnDim", outDim: "hiddenDim" }, label: "W_down" },
    ],
  },
  prenorm_attn: {
    name: "Pre-Norm + GQA + Residual", color: "#14b8a6",
    primitives: [
      { type: "rmsnorm", cfg: { dim: "hiddenDim" }, label: "Attn Norm" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_Q" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "kvDim" }, label: "W_K" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "kvDim" }, label: "W_V" },
      { type: "rope", cfg: {}, label: "RoPE" },
      { type: "scale_dot", cfg: { numHeads: "heads", headDim: "headDim" }, label: "GQA" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "hiddenDim" }, label: "W_O" },
      { type: "residual_add", cfg: {}, label: "Residual" },
    ],
  },
  prenorm_ffn: {
    name: "Pre-Norm + SwiGLU + Residual", color: "#c084fc",
    primitives: [
      { type: "rmsnorm", cfg: { dim: "hiddenDim" }, label: "FFN Norm" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "ffnDim" }, label: "W_gate" },
      { type: "silu", cfg: {}, label: "SiLU" },
      { type: "linear", cfg: { inDim: "hiddenDim", outDim: "ffnDim" }, label: "W_up" },
      { type: "gate_mul", cfg: {}, label: "Gate ⊙ Up" },
      { type: "linear", cfg: { inDim: "ffnDim", outDim: "hiddenDim" }, label: "W_down" },
      { type: "residual_add", cfg: {}, label: "Residual" },
    ],
  },
  empty: {
    name: "Empty Block", color: "#334155",
    primitives: [],
  },
};

const ARCH_PRESETS = {
  llama: {
    name: "Llama-style",
    globalCfg: { hiddenDim: 2048, ffnDim: 5504, heads: 16, kvHeads: 4, vocabSize: 32000, seqLen: 4096, moeExperts: 8, moeTopK: 2 },
    layerTemplate: [
      { template: "prenorm_attn" },
      { template: "prenorm_ffn" },
    ],
    numLayers: 16,
  },
  gpt2: {
    name: "GPT-2",
    globalCfg: { hiddenDim: 768, ffnDim: 3072, heads: 12, kvHeads: 12, vocabSize: 50257, seqLen: 1024, moeExperts: 8, moeTopK: 2 },
    layerTemplate: [
      { template: "prenorm_attn" },
      { template: "prenorm_ffn" },
    ],
    numLayers: 12,
  },
  minimal: {
    name: "Bare Minimum",
    globalCfg: { hiddenDim: 512, ffnDim: 2048, heads: 8, kvHeads: 8, vocabSize: 32000, seqLen: 2048, moeExperts: 8, moeTopK: 2 },
    layerTemplate: [
      { template: "mha" },
      { template: "ffn_swiglu" },
    ],
    numLayers: 6,
  },
};

// ─── Utilities ────────────────────────────────────────────────────────
let _id = 0;
const uid = () => `p${++_id}_${Date.now().toString(36)}`;

function fmt(n) {
  if (n >= 1e12) return (n / 1e12).toFixed(2) + "T";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toString();
}
function fmtBytes(b) {
  if (b >= 1e12) return (b / 1e12).toFixed(2) + " TB";
  if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  return (b / 1e3).toFixed(1) + " KB";
}

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

function makeLayerFromPreset(blockTemplates) {
  return {
    id: uid(), collapsed: false,
    blocks: blockTemplates.map(b => makeBlock(b.template)),
  };
}

// ─── Styles ───────────────────────────────────────────────────────────
const font = "'IBM Plex Mono', monospace";
const S = {
  panel: { background: "#0c1220", border: "1px solid #172035", borderRadius: 6 },
  label: { fontSize: 9, letterSpacing: "0.12em", textTransform: "uppercase", color: "#3e5775", fontWeight: 500, fontFamily: font },
};

// ─── Primitive Editor Row ─────────────────────────────────────────────
function PrimRow({ prim, globalCfg, onUpdate, onRemove, onMove, isFirst, isLast, dragH }) {
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
                <select value={prim.cfg[f.key]} onChange={e => {
                  const v = e.target.value;
                  updateCfg(f.key, isNaN(Number(v)) ? v : Number(v));
                }}
                  style={{ background: "#0c1220", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "1px 2px", borderRadius: 2 }}>
                  {DIM_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label} ({resolve(o.value, globalCfg)})</option>)}
                  <option value={resolve(prim.cfg[f.key], globalCfg)}>Custom: {resolve(prim.cfg[f.key], globalCfg)}</option>
                </select>
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

// ─── Primitive Palette ────────────────────────────────────────────────
function PrimPalette({ onAdd, onClose }) {
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

// ─── Component Block ──────────────────────────────────────────────────
function BlockCard({ block, globalCfg, onUpdate, onRemove, onDuplicate, onMove, isFirst, isLast }) {
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
  const insertFromTemplate = (templateKey) => {
    const t = BLOCK_TEMPLATES[templateKey];
    const newPrims = t.primitives.map(p => makePrim(p.type, p.cfg, p.label));
    onUpdate({ ...block, primitives: [...block.primitives, ...newPrims] });
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

// ─── Layer Card ───────────────────────────────────────────────────────
function LayerCard({ layer, idx, globalCfg, total, onUpdate, onRemove, onDuplicate, onMoveLayer }) {
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

// ─── Config Slider ────────────────────────────────────────────────────
function CfgSlider({ label, value, onChange, min, max, step = 1 }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 1 }}>
        <span style={S.label}>{label}</span>
        <input type="number" value={value} onChange={e => onChange(Number(e.target.value))}
          style={{ width: 60, textAlign: "right", background: "#080c16", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "1px 3px", borderRadius: 2 }} />
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: "#22c55e", height: 2 }} />
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────
export default function TransformerSimV2() {
  const [globalCfg, setGlobalCfg] = useState({
    hiddenDim: 2048, ffnDim: 5504, heads: 16, kvHeads: 4,
    vocabSize: 32000, seqLen: 4096, moeExperts: 8, moeTopK: 2,
  });
  const [layers, setLayers] = useState(() => {
    const t = ARCH_PRESETS.llama;
    return Array.from({ length: t.numLayers }, () => makeLayerFromPreset(t.layerTemplate));
  });
  const [tab, setTab] = useState("build");

  const updateCfg = (k, v) => setGlobalCfg(p => ({ ...p, [k]: v }));

  const loadPreset = (key) => {
    const p = ARCH_PRESETS[key];
    setGlobalCfg(p.globalCfg);
    setLayers(Array.from({ length: p.numLayers }, () => makeLayerFromPreset(p.layerTemplate)));
    setArchName(p.name);
    setArchDesc("");
  };

  const updateLayer = (idx, layer) => setLayers(prev => prev.map((l, i) => i === idx ? layer : l));
  const removeLayer = (idx) => setLayers(prev => prev.filter((_, i) => i !== idx));
  const duplicateLayer = (idx) => {
    const src = layers[idx];
    const dup = {
      id: uid(), collapsed: false,
      blocks: src.blocks.map(b => ({ ...b, id: uid(), primitives: b.primitives.map(p => ({ ...p, id: uid() })) })),
    };
    setLayers(prev => [...prev.slice(0, idx + 1), dup, ...prev.slice(idx + 1)]);
  };
  const moveLayer = (idx, dir) => {
    const to = idx + dir; if (to < 0 || to >= layers.length) return;
    setLayers(prev => { const n = [...prev]; [n[idx], n[to]] = [n[to], n[idx]]; return n; });
  };
  const addEmptyLayer = () => setLayers(prev => [...prev, { id: uid(), collapsed: false, blocks: [] }]);

  // Stats
  const stats = useMemo(() => {
    const embP = globalCfg.vocabSize * globalCfg.hiddenDim;
    const normP = globalCfg.hiddenDim;
    const outP = globalCfg.vocabSize * globalCfg.hiddenDim;
    let layerP = 0;
    const primCounts = {};
    const perLayer = layers.map((l, i) => {
      let lp = 0;
      l.blocks.forEach(b => b.primitives.forEach(p => {
        const reg = PRIM[p.type];
        if (!reg) return;
        const pp = reg.params(p.cfg, globalCfg);
        lp += pp;
        const k = p.type;
        if (!primCounts[k]) primCounts[k] = { count: 0, params: 0 };
        primCounts[k].count++;
        primCounts[k].params += pp;
      }));
      layerP += lp;
      return { idx: i, params: lp };
    });
    const totalP = embP + layerP + normP + outP;
    return { totalP, embP, layerP, normP, outP, perLayer, primCounts, numLayers: layers.length };
  }, [layers, globalCfg]);

  // ─── Share / Import / Export System ───────────────────────────────
  const [showShareModal, setShowShareModal] = useState(false);
  const [archName, setArchName] = useState("Untitled Architecture");
  const [archAuthor, setArchAuthor] = useState("");
  const [archDesc, setArchDesc] = useState("");
  const [toast, setToast] = useState(null);
  const [importError, setImportError] = useState(null);
  const fileInputRef = useRef(null);
  const toastTimer = useRef(null);

  const showToast = (msg, type = "success") => {
    setToast({ msg, type });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3000);
  };

  const buildExportPayload = () => ({
    _format: "vecplabs-transformer-sim",
    _version: 2,
    meta: {
      name: archName || "Untitled",
      author: archAuthor || "Anonymous",
      description: archDesc || "",
      exportedAt: new Date().toISOString(),
      totalParams: stats.totalP,
      numLayers: layers.length,
    },
    globalConfig: globalCfg,
    layers: layers.map((l, i) => ({
      layer: i + 1,
      blocks: l.blocks.map(b => ({
        name: b.name,
        color: b.color,
        primitives: b.primitives.map(p => ({
          type: p.type,
          label: p.label,
          config: p.cfg,
        })),
      })),
    })),
    stats: {
      totalParams: stats.totalP,
      embeddingParams: stats.embP,
      outputParams: stats.outP,
      layerParams: stats.layerP,
      primitiveCounts: Object.entries(stats.primCounts).map(([k, v]) => ({
        type: k, name: PRIM[k]?.name, count: v.count, totalParams: v.params,
      })),
    },
  });

  const importFromData = (data) => {
    try {
      // Validate format
      if (data._format !== "vecplabs-transformer-sim") {
        // Try legacy format (v1 export without _format tag)
        if (data.globalConfig && data.layers) {
          // Could be legacy - try anyway
        } else {
          throw new Error("Unrecognized file format. Expected a VecP Labs Transformer Simulator export.");
        }
      }

      // Import global config
      if (data.globalConfig) {
        const gc = data.globalConfig;
        setGlobalCfg({
          hiddenDim: gc.hiddenDim || 2048,
          ffnDim: gc.ffnDim || 5504,
          heads: gc.heads || 16,
          kvHeads: gc.kvHeads || 4,
          vocabSize: gc.vocabSize || 32000,
          seqLen: gc.seqLen || 4096,
          moeExperts: gc.moeExperts || 8,
          moeTopK: gc.moeTopK || 2,
        });
      }

      // Import layers
      if (data.layers && Array.isArray(data.layers)) {
        const importedLayers = data.layers.map(layerData => {
          const blocks = (layerData.blocks || []).map(blockData => {
            const primitives = (blockData.primitives || []).map(primData => {
              // Validate primitive type exists
              if (!PRIM[primData.type]) {
                console.warn(`Unknown primitive type: ${primData.type}, skipping`);
                return null;
              }
              return {
                id: uid(),
                type: primData.type,
                label: primData.label || PRIM[primData.type].name,
                cfg: primData.config || { ...PRIM[primData.type].defaultCfg },
              };
            }).filter(Boolean);

            return {
              id: uid(),
              name: blockData.name || "Imported Block",
              color: blockData.color || "#334155",
              collapsed: false,
              primitives,
            };
          });

          return { id: uid(), collapsed: false, blocks };
        });

        setLayers(importedLayers);
      }

      // Import metadata
      if (data.meta) {
        setArchName(data.meta.name || "Imported Architecture");
        setArchAuthor(data.meta.author || "");
        setArchDesc(data.meta.description || "");
      }

      setImportError(null);
      const paramStr = data.meta?.totalParams ? ` (${fmt(data.meta.totalParams)} params)` : "";
      const nameStr = data.meta?.name || "architecture";
      showToast(`Imported "${nameStr}"${paramStr}`);
      return true;
    } catch (err) {
      setImportError(err.message);
      showToast(err.message, "error");
      return false;
    }
  };

  const exportToFile = () => {
    const data = buildExportPayload();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeName = (archName || "transformer").replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
    a.download = `${safeName}-${fmt(stats.totalP)}-${layers.length}L.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Exported to file");
  };

  const exportToClipboard = async () => {
    try {
      const data = buildExportPayload();
      await navigator.clipboard.writeText(JSON.stringify(data));
      showToast("Copied to clipboard — share it!");
    } catch {
      // Fallback
      const data = buildExportPayload();
      const ta = document.createElement("textarea");
      ta.value = JSON.stringify(data);
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      showToast("Copied to clipboard — share it!");
    }
  };

  const importFromFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        if (importFromData(data)) {
          setShowShareModal(false);
        }
      } catch {
        showToast("Invalid JSON file", "error");
      }
    };
    reader.readAsText(file);
    // Reset so the same file can be re-selected
    e.target.value = "";
  };

  const importFromClipboard = async () => {
    try {
      const text = await navigator.clipboard.readText();
      const data = JSON.parse(text);
      if (importFromData(data)) {
        setShowShareModal(false);
      }
    } catch {
      showToast("No valid config found on clipboard", "error");
    }
  };

  // ─── Share Modal ──────────────────────────────────────────────
  const ShareModal = () => (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(0,0,0,0.7)", backdropFilter: "blur(4px)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 16,
    }} onClick={() => setShowShareModal(false)}>
      <div style={{
        ...S.panel, width: "100%", maxWidth: 420, maxHeight: "90vh", overflowY: "auto",
        padding: 0, background: "#0a0f1c", border: "1px solid #22c55e22",
        boxShadow: "0 0 40px rgba(34,197,94,0.08)",
      }} onClick={e => e.stopPropagation()}>
        {/* Modal header */}
        <div style={{
          padding: "12px 14px", borderBottom: "1px solid #172035",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <div style={{ fontSize: 7, color: "#22c55e", letterSpacing: "0.3em", textTransform: "uppercase", fontWeight: 600 }}>Share & Import</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2eaf2", marginTop: 2 }}>Architecture Config</div>
          </div>
          <button onClick={() => setShowShareModal(false)} style={{
            background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 16, fontWeight: 700, padding: "0 4px",
          }}>×</button>
        </div>

        {/* Metadata fields */}
        <div style={{ padding: "10px 14px", borderBottom: "1px solid #172035" }}>
          <div style={{ ...S.label, marginBottom: 6, color: "#06b6d4" }}>Architecture Metadata</div>
          <div style={{ marginBottom: 6 }}>
            <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Name</label>
            <input value={archName} onChange={e => setArchName(e.target.value)}
              placeholder="e.g. WobbleNet v3, Cerberus 56M Scout..."
              style={{
                width: "100%", background: "#060a14", border: "1px solid #172035", color: "#c8d6e5",
                fontFamily: font, fontSize: 10, padding: "5px 8px", borderRadius: 3,
              }} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 6 }}>
            <div>
              <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Author</label>
              <input value={archAuthor} onChange={e => setArchAuthor(e.target.value)}
                placeholder="Your name"
                style={{
                  width: "100%", background: "#060a14", border: "1px solid #172035", color: "#c8d6e5",
                  fontFamily: font, fontSize: 10, padding: "5px 8px", borderRadius: 3,
                }} />
            </div>
            <div>
              <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Params</label>
              <div style={{
                background: "#060a14", border: "1px solid #172035", borderRadius: 3,
                padding: "5px 8px", fontSize: 10, color: "#22c55e", fontWeight: 600,
              }}>{fmt(stats.totalP)} ({stats.numLayers}L)</div>
            </div>
          </div>
          <div>
            <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Notes</label>
            <textarea value={archDesc} onChange={e => setArchDesc(e.target.value)} rows={2}
              placeholder="Design notes, rationale, what you changed..."
              style={{
                width: "100%", background: "#060a14", border: "1px solid #172035", color: "#c8d6e5",
                fontFamily: font, fontSize: 10, padding: "5px 8px", borderRadius: 3, resize: "vertical",
              }} />
          </div>
        </div>

        {/* Export section */}
        <div style={{ padding: "10px 14px", borderBottom: "1px solid #172035" }}>
          <div style={{ ...S.label, marginBottom: 8, color: "#22c55e" }}>Export</div>
          <div style={{ display: "flex", gap: 6 }}>
            <button onClick={exportToFile} style={{
              flex: 1, background: "linear-gradient(135deg, #22c55e, #16a34a)", color: "#000",
              border: "none", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 700, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ↓ Save File
            </button>
            <button onClick={exportToClipboard} style={{
              flex: 1, background: "linear-gradient(135deg, #06b6d4, #0891b2)", color: "#000",
              border: "none", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 700, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ⧉ Copy to Clipboard
            </button>
          </div>
          <div style={{ fontSize: 8, color: "#2a3f55", marginTop: 4, textAlign: "center" }}>
            Copy → paste in chat/email/DM → recipient imports
          </div>
        </div>

        {/* Import section */}
        <div style={{ padding: "10px 14px" }}>
          <div style={{ ...S.label, marginBottom: 8, color: "#f97316" }}>Import</div>
          <div style={{ display: "flex", gap: 6 }}>
            <button onClick={() => fileInputRef.current?.click()} style={{
              flex: 1, background: "#172035", color: "#c8d6e5",
              border: "1px solid #22354a", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 600, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ↑ Load File
            </button>
            <button onClick={importFromClipboard} style={{
              flex: 1, background: "#172035", color: "#c8d6e5",
              border: "1px solid #22354a", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 600, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ⧉ Paste from Clipboard
            </button>
          </div>
          <input ref={fileInputRef} type="file" accept=".json,application/json" onChange={importFromFile}
            style={{ display: "none" }} />
          {importError && (
            <div style={{ fontSize: 9, color: "#ef4444", marginTop: 6, padding: "4px 8px", background: "#1a0a0a", borderRadius: 3, border: "1px solid #2a1515" }}>
              {importError}
            </div>
          )}
          <div style={{ fontSize: 8, color: "#2a3f55", marginTop: 4, textAlign: "center" }}>
            Import replaces current architecture. Export first to save your work.
          </div>
        </div>
      </div>
    </div>
  );

  // ─── Toast Notification ───────────────────────────────────────
  const Toast = () => {
    if (!toast) return null;
    const isErr = toast.type === "error";
    return (
      <div style={{
        position: "fixed", bottom: 20, left: "50%", transform: "translateX(-50%)",
        zIndex: 1001, padding: "8px 16px", borderRadius: 6,
        background: isErr ? "#1a0a0a" : "#0a1a0a",
        border: `1px solid ${isErr ? "#ef444433" : "#22c55e33"}`,
        color: isErr ? "#ef4444" : "#22c55e",
        fontFamily: font, fontSize: 10, fontWeight: 600,
        boxShadow: `0 4px 20px ${isErr ? "rgba(239,68,68,0.15)" : "rgba(34,197,94,0.15)"}`,
        animation: "toastIn 0.2s ease",
        maxWidth: "90vw", textAlign: "center",
      }}>
        {isErr ? "✕ " : "✓ "}{toast.msg}
      </div>
    );
  };

  return (
    <div style={{
      background: "#060a14", color: "#c8d6e5", minHeight: "100vh", fontFamily: font,
      backgroundImage: "linear-gradient(rgba(34,197,94,0.015) 1px, transparent 1px), linear-gradient(90deg, rgba(34,197,94,0.015) 1px, transparent 1px)",
      backgroundSize: "20px 20px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      <style>{`
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: #172035; border-radius: 2px; }
        input[type="range"] { height: 2px; }
        * { box-sizing: border-box; }
        @keyframes toastIn {
          from { opacity: 0; transform: translateX(-50%) translateY(10px); }
          to { opacity: 1; transform: translateX(-50%) translateY(0); }
        }
      `}</style>

      {/* Toast */}
      <Toast />

      {/* Share Modal */}
      {showShareModal && <ShareModal />}

      {/* Header */}
      <div style={{ padding: "14px 14px 0" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: 7, color: "#22c55e", letterSpacing: "0.4em", textTransform: "uppercase", fontWeight: 600 }}>VecP Labs</div>
            <h1 style={{ fontSize: 16, fontWeight: 700, color: "#e2eaf2", margin: "2px 0 0", letterSpacing: "-0.02em" }}>
              Transformer Simulator <span style={{ fontSize: 9, color: "#3e5775", fontWeight: 400 }}>v2</span>
            </h1>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <button onClick={() => setShowShareModal(true)} style={{
              background: "linear-gradient(135deg, #22c55e, #16a34a)", color: "#000",
              border: "none", borderRadius: 4, padding: "5px 10px", fontSize: 8,
              fontFamily: font, fontWeight: 700, cursor: "pointer", letterSpacing: "0.1em", textTransform: "uppercase",
            }}>Share</button>
          </div>
        </div>

        {/* Architecture name display */}
        <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 4 }}>
          <span style={{ fontSize: 11, color: "#6b8299", fontWeight: 500 }}>{archName}</span>
          {archAuthor && <span style={{ fontSize: 8, color: "#2a3f55" }}>by {archAuthor}</span>}
        </div>

        {/* Summary */}
        <div style={{ display: "flex", gap: 5, marginTop: 10, marginBottom: 8, flexWrap: "wrap" }}>
          {[
            { label: "Total", value: fmt(stats.totalP), color: "#22c55e" },
            { label: "Layers", value: stats.numLayers, color: "#06b6d4" },
            { label: "F16", value: fmtBytes(stats.totalP * 2), color: "#a855f7" },
            { label: "Q4", value: fmtBytes(stats.totalP * 0.5625), color: "#ec4899" },
          ].map((s, i) => (
            <div key={i} style={{ ...S.panel, padding: "5px 8px", flex: 1, minWidth: 70, borderLeft: `2px solid ${s.color}` }}>
              <div style={{ fontSize: 7, color: "#3e5775", textTransform: "uppercase", letterSpacing: "0.1em" }}>{s.label}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: s.color, marginTop: 1 }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 2, marginBottom: 6 }}>
          {[["build", "Build"], ["stats", "Stats"]].map(([k, l]) => (
            <button key={k} onClick={() => setTab(k)} style={{
              flex: 1, padding: "5px 0", fontSize: 8, fontFamily: font, cursor: "pointer",
              textTransform: "uppercase", letterSpacing: "0.1em",
              background: tab === k ? "#0c1220" : "transparent",
              border: tab === k ? "1px solid #172035" : "1px solid transparent",
              borderBottom: tab === k ? "none" : "1px solid #172035",
              color: tab === k ? "#c8d6e5" : "#2a3f55", borderRadius: "4px 4px 0 0",
            }}>{l}</button>
          ))}
        </div>
      </div>

      {tab === "build" && (
        <div style={{ padding: "0 14px 14px" }}>
          {/* Presets */}
          <div style={{ ...S.panel, padding: "6px 8px", marginBottom: 8 }}>
            <div style={{ ...S.label, marginBottom: 4 }}>Presets</div>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              {Object.entries(ARCH_PRESETS).map(([k, p]) => (
                <button key={k} onClick={() => loadPreset(k)} style={{
                  background: "#080c16", border: "1px solid #172035", color: "#6b8299",
                  borderRadius: 3, padding: "3px 7px", fontSize: 8, cursor: "pointer", fontFamily: font,
                }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = "#22c55e33"; e.currentTarget.style.color = "#22c55e"; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = "#172035"; e.currentTarget.style.color = "#6b8299"; }}
                >{p.name}</button>
              ))}
            </div>
          </div>

          {/* Global Config */}
          <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
            <div style={{ ...S.label, marginBottom: 6, color: "#22c55e" }}>Global Dimensions</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 12px" }}>
              <CfgSlider label="Hidden Dim" value={globalCfg.hiddenDim} onChange={v => updateCfg("hiddenDim", v)} min={64} max={8192} step={64} />
              <CfgSlider label="FFN Dim" value={globalCfg.ffnDim} onChange={v => updateCfg("ffnDim", v)} min={64} max={32768} step={64} />
              <CfgSlider label="Attn Heads" value={globalCfg.heads} onChange={v => updateCfg("heads", v)} min={1} max={64} />
              <CfgSlider label="KV Heads" value={globalCfg.kvHeads} onChange={v => updateCfg("kvHeads", v)} min={1} max={globalCfg.heads} />
              <CfgSlider label="Vocab" value={globalCfg.vocabSize} onChange={v => updateCfg("vocabSize", v)} min={256} max={256000} step={256} />
              <CfgSlider label="Seq Len" value={globalCfg.seqLen} onChange={v => updateCfg("seqLen", v)} min={128} max={131072} step={128} />
              <CfgSlider label="MoE Experts" value={globalCfg.moeExperts} onChange={v => updateCfg("moeExperts", v)} min={2} max={64} />
              <CfgSlider label="MoE Top-K" value={globalCfg.moeTopK} onChange={v => updateCfg("moeTopK", v)} min={1} max={globalCfg.moeExperts} />
            </div>
          </div>

          {/* Embedding */}
          <div style={{ ...S.panel, padding: "5px 8px", marginBottom: 3, borderLeft: "3px solid #f97316", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 10, color: "#f97316" }}>◈ Token Embedding <span style={{ color: "#2a3f55", fontSize: 8 }}>[{globalCfg.vocabSize} → {globalCfg.hiddenDim}]</span></span>
            <span style={{ fontSize: 9, color: "#22c55e", fontWeight: 600 }}>{fmt(stats.embP)}</span>
          </div>
          <div style={{ textAlign: "center", color: "#111827", fontSize: 8, margin: "1px 0" }}>│</div>

          {/* Layers */}
          {layers.map((layer, idx) => (
            <div key={layer.id}>
              <LayerCard
                layer={layer} idx={idx} globalCfg={globalCfg} total={layers.length}
                onUpdate={l => updateLayer(idx, l)}
                onRemove={() => removeLayer(idx)}
                onDuplicate={() => duplicateLayer(idx)}
                onMoveLayer={dir => moveLayer(idx, dir)}
              />
              {idx < layers.length - 1 && <div style={{ textAlign: "center", color: "#111827", fontSize: 8, margin: "-2px 0" }}>│</div>}
            </div>
          ))}

          <div style={{ textAlign: "center", color: "#111827", fontSize: 8, margin: "1px 0" }}>│</div>
          <div style={{ ...S.panel, padding: "5px 8px", marginBottom: 6, borderLeft: "3px solid #ef4444", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 10, color: "#ef4444" }}>◈ Final Norm + LM Head <span style={{ color: "#2a3f55", fontSize: 8 }}>[{globalCfg.hiddenDim} → {globalCfg.vocabSize}]</span></span>
            <span style={{ fontSize: 9, color: "#22c55e", fontWeight: 600 }}>{fmt(stats.outP + stats.normP)}</span>
          </div>

          <button onClick={addEmptyLayer} style={{
            width: "100%", ...S.panel, padding: "7px 0", border: "1px dashed #172035",
            background: "transparent", color: "#2a3f55", fontSize: 9, cursor: "pointer", fontFamily: font,
          }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = "#22c55e33"; e.currentTarget.style.color = "#22c55e"; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = "#172035"; e.currentTarget.style.color = "#2a3f55"; }}
          >+ Add Layer</button>
        </div>
      )}

      {tab === "stats" && (
        <div style={{ padding: "0 14px 14px" }}>
          {/* Primitive type breakdown */}
          <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
            <div style={{ ...S.label, marginBottom: 6, color: "#22c55e" }}>Primitive Breakdown (across all layers)</div>
            {Object.entries(stats.primCounts)
              .sort((a, b) => b[1].params - a[1].params)
              .map(([k, data]) => {
                const reg = PRIM[k];
                if (!reg) return null;
                const pct = stats.layerP > 0 ? (data.params / stats.layerP * 100) : 0;
                return (
                  <div key={k} style={{ marginBottom: 6 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 2 }}>
                      <span style={{ fontSize: 9, color: reg.color, fontWeight: 600 }}>{reg.icon} {reg.name}</span>
                      <span style={{ fontSize: 8, color: "#3e5775" }}>
                        ×{data.count} = <span style={{ color: "#22c55e" }}>{fmt(data.params)}</span>
                        {pct > 0 && <span style={{ color: "#1a2744", marginLeft: 4 }}>{pct.toFixed(1)}%</span>}
                      </span>
                    </div>
                    {data.params > 0 && (
                      <div style={{ height: 3, background: "#080c16", borderRadius: 2, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${pct}%`, background: reg.color, borderRadius: 2, transition: "width 0.3s" }} />
                      </div>
                    )}
                  </div>
                );
              })}
          </div>

          {/* Fixed costs */}
          <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
            <div style={{ ...S.label, marginBottom: 6, color: "#f97316" }}>Fixed Costs</div>
            {[
              { name: "Token Embedding", params: stats.embP, color: "#f97316" },
              { name: "Final RMSNorm", params: stats.normP, color: "#22c55e" },
              { name: "LM Head", params: stats.outP, color: "#ef4444" },
            ].map(item => (
              <div key={item.name} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid #080c16" }}>
                <span style={{ fontSize: 9, color: item.color }}>{item.name}</span>
                <span style={{ fontSize: 9, color: "#8899aa", fontWeight: 600 }}>{fmt(item.params)}</span>
              </div>
            ))}
          </div>

          {/* Per-layer */}
          <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
            <div style={{ ...S.label, marginBottom: 6, color: "#06b6d4" }}>Per-Layer Parameters</div>
            <div style={{ maxHeight: 180, overflowY: "auto" }}>
              {stats.perLayer.map((ls, i) => {
                const maxP = Math.max(...stats.perLayer.map(s => s.params), 1);
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                    <span style={{ fontSize: 8, color: "#2a3f55", width: 20, textAlign: "right", flexShrink: 0 }}>L{i + 1}</span>
                    <div style={{ flex: 1, height: 8, background: "#080c16", borderRadius: 2, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${ls.params / maxP * 100}%`, background: "linear-gradient(90deg, #06b6d4, #22c55e)", borderRadius: 2 }} />
                    </div>
                    <span style={{ fontSize: 8, color: "#6b8299", width: 45, textAlign: "right", flexShrink: 0 }}>{fmt(ls.params)}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Fingerprint */}
          <div style={{ ...S.panel, padding: "8px 10px" }}>
            <div style={{ ...S.label, marginBottom: 6, color: "#a855f7" }}>Architecture Fingerprint</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {[
                { l: "FFN / Hidden", v: `${(globalCfg.ffnDim / globalCfg.hiddenDim).toFixed(2)}×` },
                { l: "Head Dim", v: Math.floor(globalCfg.hiddenDim / globalCfg.heads) },
                { l: "Q/KV Ratio", v: `${globalCfg.heads}/${globalCfg.kvHeads}` },
                { l: "Params/Layer", v: fmt(stats.numLayers > 0 ? stats.layerP / stats.numLayers : 0) },
                { l: "Chinchilla", v: fmt(stats.totalP * 20) },
                { l: "Layer Params %", v: `${(stats.layerP / stats.totalP * 100).toFixed(1)}%` },
              ].map((item, i) => (
                <div key={i} style={{ background: "#080c16", borderRadius: 3, padding: "5px 7px" }}>
                  <div style={{ fontSize: 7, color: "#2a3f55", textTransform: "uppercase" }}>{item.l}</div>
                  <div style={{ fontSize: 12, color: "#a0b4c8", fontWeight: 600, marginTop: 1 }}>{item.v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div style={{ textAlign: "center", padding: "8px 0 16px", fontSize: 7, color: "#111827" }}>
        Drag primitives to reorder · Double-click block names to rename · ⚙ to configure · Share to export/import configs
      </div>
    </div>
  );
}
