import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { PRIM, computeBlockFlops } from "./data/primitives";
import { BLOCK_TEMPLATES } from "./data/blockTemplates";
import { ARCH_PRESETS } from "./data/presets";
import { resolve } from "./data/dimensions";
import { uid } from "./utils/uid";
import { fmt, fmtBytes } from "./utils/format";
import { S, font, THEMES } from "./components/styles";
import HelpOverlay from "./components/HelpOverlay";
import LayerCard from "./components/LayerCard";
import ConfigPanel from "./components/ConfigPanel";
import StatsPanel from "./components/StatsPanel";
import ShareModal from "./components/ShareModal";
import Toast from "./components/Toast";
import DiffView from "./components/DiffView";
import PyTorchExport from "./components/PyTorchExport";
import BatchOps from "./components/BatchOps";
import ScatterPlot from "./components/ScatterPlot";
import { buildExportPayload, parseImportData } from "./utils/exportImport";

// ─── Factory helpers ──────────────────────────────────────────────────
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

// ─── Undo / Redo ──────────────────────────────────────────────────────
const MAX_HISTORY = 50;

function useHistory(initialLayers, initialCfg) {
  const [layers, setLayersRaw] = useState(initialLayers);
  const [globalCfg, setGlobalCfgRaw] = useState(initialCfg);
  const history = useRef([{ layers: initialLayers, globalCfg: initialCfg }]);
  const pointer = useRef(0);
  const skipRecord = useRef(false);

  const record = useCallback((newLayers, newCfg) => {
    if (skipRecord.current) { skipRecord.current = false; return; }
    const next = { layers: newLayers, globalCfg: newCfg };
    history.current = [...history.current.slice(0, pointer.current + 1), next].slice(-MAX_HISTORY);
    pointer.current = history.current.length - 1;
  }, []);

  const setLayers = useCallback((updater) => {
    setLayersRaw(prev => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      // defer record to after globalCfg is also settled
      setTimeout(() => {
        setGlobalCfgRaw(cfg => { record(next, cfg); return cfg; });
      }, 0);
      return next;
    });
  }, [record]);

  const setGlobalCfg = useCallback((updater) => {
    setGlobalCfgRaw(prev => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      setTimeout(() => {
        setLayersRaw(l => { record(l, next); return l; });
      }, 0);
      return next;
    });
  }, [record]);

  // Bulk set for imports / presets (single history entry)
  const setBoth = useCallback((newLayers, newCfg) => {
    skipRecord.current = true;
    setLayersRaw(newLayers);
    skipRecord.current = true;
    setGlobalCfgRaw(newCfg);
    const next = { layers: newLayers, globalCfg: newCfg };
    history.current = [...history.current.slice(0, pointer.current + 1), next].slice(-MAX_HISTORY);
    pointer.current = history.current.length - 1;
  }, []);

  const undo = useCallback(() => {
    if (pointer.current <= 0) return;
    pointer.current--;
    const snap = history.current[pointer.current];
    skipRecord.current = true;
    setLayersRaw(snap.layers);
    skipRecord.current = true;
    setGlobalCfgRaw(snap.globalCfg);
  }, []);

  const redo = useCallback(() => {
    if (pointer.current >= history.current.length - 1) return;
    pointer.current++;
    const snap = history.current[pointer.current];
    skipRecord.current = true;
    setLayersRaw(snap.layers);
    skipRecord.current = true;
    setGlobalCfgRaw(snap.globalCfg);
  }, []);

  const canUndo = pointer.current > 0;
  const canRedo = pointer.current < history.current.length - 1;

  return { layers, globalCfg, setLayers, setGlobalCfg, setBoth, undo, redo, canUndo, canRedo, historyRef: history, pointerRef: pointer };
}

// ─── Main ─────────────────────────────────────────────────────────────
export default function TransformerSimV2() {
  const initPreset = ARCH_PRESETS.llama;
  const initLayers = Array.from({ length: initPreset.numLayers }, () => makeLayerFromPreset(initPreset.layerTemplate));

  const {
    layers, globalCfg, setLayers, setGlobalCfg, setBoth,
    undo, redo, canUndo, canRedo, historyRef, pointerRef,
  } = useHistory(initLayers, initPreset.globalCfg);

  const [tab, setTab] = useState("build");
  const [showShareModal, setShowShareModal] = useState(false);
  const [archName, setArchName] = useState("Untitled Architecture");
  const [archAuthor, setArchAuthor] = useState("");
  const [archDesc, setArchDesc] = useState("");
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);
  const [budget, setBudget] = useState(null); // null = off, number = target params

  // Theme
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("tsim-theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });
  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("tsim-theme", next);
  };

  // Help / Onboarding
  const [showHelp, setShowHelp] = useState(() => !localStorage.getItem("tsim-seen-help"));
  const dismissHelp = () => { setShowHelp(false); localStorage.setItem("tsim-seen-help", "1"); };

  const showToast = useCallback((msg, type = "success") => {
    setToast({ msg, type });
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3000);
  }, []);

  const updateCfg = (k, v) => setGlobalCfg(p => ({ ...p, [k]: v }));

  const loadPreset = (key) => {
    const p = ARCH_PRESETS[key];
    const newLayers = Array.from({ length: p.numLayers }, () => makeLayerFromPreset(p.layerTemplate));
    setBoth(newLayers, p.globalCfg);
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

  // Collapse All / Expand All
  const allCollapsed = layers.length > 0 && layers.every(l => l.collapsed);
  const toggleCollapseAll = () => {
    const newCollapsed = !allCollapsed;
    setLayers(prev => prev.map(l => ({
      ...l, collapsed: newCollapsed,
      blocks: l.blocks.map(b => ({ ...b, collapsed: newCollapsed })),
    })));
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
      const mod = isMac ? e.metaKey : e.ctrlKey;
      if (mod && e.key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      if (mod && (e.key === "y" || (e.key === "z" && e.shiftKey))) { e.preventDefault(); redo(); }
      if (mod && e.key === "s") { e.preventDefault(); setShowShareModal(true); }
      if (mod && e.shiftKey && (e.key === "E" || e.key === "e")) {
        e.preventDefault();
        setLayers(prev => prev.map(l => ({ ...l, collapsed: false, blocks: l.blocks.map(b => ({ ...b, collapsed: false })) })));
      }
      if (mod && e.shiftKey && (e.key === "C" || e.key === "c")) {
        e.preventDefault();
        setLayers(prev => prev.map(l => ({ ...l, collapsed: true, blocks: l.blocks.map(b => ({ ...b, collapsed: true })) })));
      }
      if (e.key === "?" && !e.ctrlKey && !e.metaKey && e.target.tagName !== "INPUT" && e.target.tagName !== "TEXTAREA" && e.target.tagName !== "SELECT") {
        e.preventDefault();
        setShowHelp(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [undo, redo, setLayers]);

  // Stats
  const stats = useMemo(() => {
    const embP = globalCfg.vocabSize * globalCfg.hiddenDim;
    const normP = globalCfg.hiddenDim;
    const outP = globalCfg.vocabSize * globalCfg.hiddenDim;
    let layerP = 0;
    let totalFlops = 0;
    const primCounts = {};
    const perLayer = layers.map((l, i) => {
      let lp = 0;
      let lf = 0;
      l.blocks.forEach(b => {
        lf += computeBlockFlops(b.primitives, globalCfg);
        b.primitives.forEach(p => {
          const reg = PRIM[p.type];
          if (!reg) return;
          const pp = reg.params(p.cfg, globalCfg);
          lp += pp;
          const k = p.type;
          if (!primCounts[k]) primCounts[k] = { count: 0, params: 0 };
          primCounts[k].count++;
          primCounts[k].params += pp;
        });
      });
      layerP += lp;
      totalFlops += lf;
      return { idx: i, params: lp, flops: lf };
    });
    // Fixed FLOPs: embedding lookup (~0), final norm, LM head projection
    const fixedFlops = 5 * globalCfg.hiddenDim + 2 * globalCfg.hiddenDim * globalCfg.vocabSize;
    totalFlops += fixedFlops;
    const totalP = embP + layerP + normP + outP;
    return { totalP, embP, layerP, normP, outP, perLayer, primCounts, numLayers: layers.length, totalFlops, fixedFlops };
  }, [layers, globalCfg]);

  // Import handler
  const handleImport = useCallback((result) => {
    setBoth(result.layers, result.globalCfg);
    setArchName(result.meta.name);
    setArchAuthor(result.meta.author);
    setArchDesc(result.meta.desc);
  }, [setBoth]);

  // URL hash import on mount
  useEffect(() => {
    try {
      const hash = window.location.hash;
      if (!hash.startsWith("#config=")) return;
      const encoded = hash.slice(8);
      const json = decodeURIComponent(atob(encoded));
      const data = JSON.parse(json);
      const result = parseImportData(data);
      setBoth(result.layers, result.globalCfg);
      setArchName(result.meta.name);
      setArchAuthor(result.meta.author);
      setArchDesc(result.meta.desc);
      // Clean hash without triggering reload
      history.replaceState(null, "", window.location.pathname + window.location.search);
      showToast(`Loaded "${result.meta.name}" from link`);
    } catch {}
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Apply theme CSS variables
  const themeVars = THEMES[theme] || THEMES.dark;

  return (
    <div style={{
      ...themeVars,
      background: "var(--bg)", color: "var(--text)", minHeight: "100vh", fontFamily: font,
      backgroundImage: `linear-gradient(var(--bg-grid) 1px, transparent 1px), linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px)`,
      backgroundSize: "20px 20px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      <style>{`
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: var(--scrollbar, #172035); border-radius: 2px; }
        input[type="range"] { height: 2px; }
        * { box-sizing: border-box; }
        @keyframes toastIn {
          from { opacity: 0; transform: translateX(-50%) translateY(10px); }
          to { opacity: 1; transform: translateX(-50%) translateY(0); }
        }
        @media (max-width: 480px) {
          input[type="range"] { height: 6px; }
          input[type="range"]::-webkit-slider-thumb { width: 18px; height: 18px; }
          button { min-height: 32px; }
        }
      `}</style>

      {/* Help Overlay */}
      {showHelp && <HelpOverlay onClose={dismissHelp} />}

      {/* Toast */}
      <Toast toast={toast} />

      {/* Share Modal */}
      {showShareModal && (
        <ShareModal
          archName={archName} setArchName={setArchName}
          archAuthor={archAuthor} setArchAuthor={setArchAuthor}
          archDesc={archDesc} setArchDesc={setArchDesc}
          stats={stats} globalCfg={globalCfg} layers={layers}
          onImport={handleImport} onClose={() => setShowShareModal(false)} showToast={showToast}
        />
      )}

      {/* Header */}
      <div style={{ padding: "14px 14px 0" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: 7, color: "var(--accent-green)", letterSpacing: "0.4em", textTransform: "uppercase", fontWeight: 600 }}>VecP Labs</div>
            <h1 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-bright)", margin: "2px 0 0", letterSpacing: "-0.02em" }}>
              Transformer Simulator <span style={{ fontSize: 9, color: "var(--muted)", fontWeight: 400 }}>v2</span>
            </h1>
          </div>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <button onClick={() => setShowHelp(true)} title="Help (?)" style={{
              background: "none", border: "1px solid var(--border)", borderRadius: 3, padding: "4px 7px",
              color: "var(--muted)", cursor: "pointer", fontSize: 9, fontFamily: font, fontWeight: 700,
            }}>?</button>
            <button onClick={toggleTheme} title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} style={{
              background: "none", border: "1px solid var(--border)", borderRadius: 3, padding: "4px 7px",
              color: "var(--muted)", cursor: "pointer", fontSize: 9, fontFamily: font,
            }}>{theme === "dark" ? "☀" : "☾"}</button>
            <button onClick={undo} disabled={!canUndo} title="Undo (Ctrl+Z)" style={{
              background: "none", border: "1px solid var(--border)", borderRadius: 3, padding: "4px 7px",
              color: canUndo ? "var(--text-secondary)" : "var(--muted-faint)", cursor: canUndo ? "pointer" : "default",
              fontSize: 9, fontFamily: font,
            }}>↩</button>
            <button onClick={redo} disabled={!canRedo} title="Redo (Ctrl+Y)" style={{
              background: "none", border: "1px solid var(--border)", borderRadius: 3, padding: "4px 7px",
              color: canRedo ? "var(--text-secondary)" : "var(--muted-faint)", cursor: canRedo ? "pointer" : "default",
              fontSize: 9, fontFamily: font,
            }}>↪</button>
            <button onClick={() => setShowShareModal(true)} title="Share (Ctrl+S)" style={{
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
        <div style={{ display: "flex", gap: 5, marginTop: 10, marginBottom: budget !== null ? 4 : 8, flexWrap: "wrap" }}>
          {[
            { label: "Total", value: fmt(stats.totalP), color: "#22c55e" },
            { label: "FLOPs/tok", value: fmt(stats.totalFlops), color: "#06b6d4" },
            { label: "F16", value: fmtBytes(stats.totalP * 2), color: "#a855f7" },
            { label: "Q4", value: fmtBytes(stats.totalP * 0.5625), color: "#ec4899" },
          ].map((s, i) => (
            <div key={i} style={{ ...S.panel, padding: "5px 8px", flex: 1, minWidth: 70, borderLeft: `2px solid ${s.color}` }}>
              <div style={{ fontSize: 7, color: "#3e5775", textTransform: "uppercase", letterSpacing: "0.1em" }}>{s.label}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color: s.color, marginTop: 1 }}>{s.value}</div>
            </div>
          ))}
        </div>

        {/* Budget bar */}
        {budget !== null && (() => {
          const pct = budget > 0 ? (stats.totalP / budget * 100) : 0;
          const delta = stats.totalP - budget;
          const barColor = pct > 100 ? "#ef4444" : pct > 80 ? "#f59e0b" : "#22c55e";
          return (
            <div style={{ ...S.panel, padding: "5px 8px", marginBottom: 8, borderLeft: `2px solid ${barColor}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ fontSize: 7, color: "#3e5775", textTransform: "uppercase", letterSpacing: "0.1em" }}>Budget</span>
                  <input type="number" value={budget / 1e6} onChange={e => setBudget(Math.max(0, Number(e.target.value)) * 1e6)}
                    style={{ width: 50, background: "#080c16", border: "1px solid #172035", color: "#a0b4c8", fontFamily: font, fontSize: 9, padding: "1px 3px", borderRadius: 2, textAlign: "right" }} />
                  <span style={{ fontSize: 8, color: "#3e5775" }}>M</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 9, fontWeight: 600, color: barColor }}>
                    {delta > 0 ? `+${fmt(delta)} over` : `${fmt(Math.abs(delta))} remaining`}
                  </span>
                  <button onClick={() => setBudget(null)} style={{ background: "none", border: "none", color: "#3a2020", cursor: "pointer", fontSize: 9, fontWeight: 700, padding: 0 }}>×</button>
                </div>
              </div>
              <div style={{ height: 4, background: "#080c16", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${Math.min(pct, 100)}%`, background: barColor, borderRadius: 2, transition: "width 0.3s, background 0.3s" }} />
              </div>
              <div style={{ fontSize: 7, color: "#2a3f55", textAlign: "right", marginTop: 1 }}>{pct.toFixed(1)}%</div>
            </div>
          );
        })()}
        {budget === null && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 4 }}>
            <button onClick={() => setBudget(Math.round(stats.totalP / 1e6) * 1e6 || 500e6)} style={{
              background: "none", border: "1px solid #172035", borderRadius: 3,
              color: "#2a3f55", cursor: "pointer", fontSize: 7, fontFamily: font,
              padding: "2px 6px", letterSpacing: "0.05em", textTransform: "uppercase",
            }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = "#22c55e33"; e.currentTarget.style.color = "#22c55e"; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = "#172035"; e.currentTarget.style.color = "#2a3f55"; }}
            >+ Set Budget</button>
          </div>
        )}

        {/* Tabs */}
        <div style={{ display: "flex", gap: 2, marginBottom: 6 }}>
          {[["build", "Build"], ["stats", "Stats"], ["diff", "Diff"], ["code", "Code"]].map(([k, l]) => (
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
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <div style={S.label}>Presets</div>
              <button onClick={toggleCollapseAll} style={{
                background: "none", border: "1px solid #172035", borderRadius: 3,
                color: "#6b8299", cursor: "pointer", fontSize: 7, fontFamily: font,
                padding: "2px 6px", letterSpacing: "0.05em", textTransform: "uppercase",
              }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = "#22c55e33"; e.currentTarget.style.color = "#22c55e"; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = "#172035"; e.currentTarget.style.color = "#6b8299"; }}
              >{allCollapsed ? "Expand All" : "Collapse All"}</button>
            </div>
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

          {/* Batch Operations */}
          <BatchOps layers={layers} onUpdateLayers={newLayers => setLayers(newLayers)} showToast={showToast} />

          {/* Global Config */}
          <ConfigPanel globalCfg={globalCfg} updateCfg={updateCfg} />

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
        <div>
          <StatsPanel stats={stats} globalCfg={globalCfg} />
          <div style={{ padding: "0 14px 14px" }}>
            <ScatterPlot stats={stats} />
          </div>
        </div>
      )}

      {tab === "diff" && (
        <DiffView
          currentLayers={layers} currentCfg={globalCfg} currentName={archName}
          showToast={showToast} onClose={() => setTab("build")}
        />
      )}

      {tab === "code" && (
        <div style={{ padding: "0 14px 14px" }}>
          <PyTorchExport layers={layers} globalCfg={globalCfg} archName={archName} showToast={showToast} />
        </div>
      )}

      <div style={{ textAlign: "center", padding: "8px 14px 16px", fontSize: 7, color: "var(--muted-faint)", lineHeight: 1.8 }}>
        Ctrl+Z/Y undo/redo · Ctrl+S share · Ctrl+Shift+E expand · Ctrl+Shift+C collapse · ? help
      </div>
    </div>
  );
}
