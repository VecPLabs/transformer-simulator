import { useState, useRef, useMemo } from "react";
import { PRIM } from "../data/primitives";
import { resolve } from "../data/dimensions";
import { fmt } from "../utils/format";
import { parseImportData } from "../utils/exportImport";
import { S, font } from "./styles";

function computeStats(layers, globalCfg) {
  let totalP = globalCfg.vocabSize * globalCfg.hiddenDim * 2 + globalCfg.hiddenDim; // emb + out + norm
  layers.forEach(l => l.blocks.forEach(b => b.primitives.forEach(p => {
    const reg = PRIM[p.type];
    if (reg) totalP += reg.params(p.cfg, globalCfg);
  })));
  return totalP;
}

function flattenPrims(layers) {
  const result = [];
  layers.forEach((l, li) => l.blocks.forEach((b, bi) => b.primitives.forEach((p, pi) => {
    result.push({ layerIdx: li, blockName: b.name, prim: p });
  })));
  return result;
}

export default function DiffView({ currentLayers, currentCfg, currentName, showToast, onClose }) {
  const [compareData, setCompareData] = useState(null);
  const fileRef = useRef(null);

  const loadFromFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target.result);
        const result = parseImportData(data);
        setCompareData(result);
      } catch {
        showToast("Invalid JSON file", "error");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const loadFromClipboard = async () => {
    try {
      const text = await navigator.clipboard.readText();
      const data = JSON.parse(text);
      const result = parseImportData(data);
      setCompareData(result);
    } catch {
      showToast("No valid config on clipboard", "error");
    }
  };

  const diff = useMemo(() => {
    if (!compareData) return null;
    const aLayers = currentLayers;
    const bLayers = compareData.layers;
    const aCfg = currentCfg;
    const bCfg = compareData.globalCfg;

    const aParams = computeStats(aLayers, aCfg);
    const bParams = computeStats(bLayers, bCfg);

    const maxLayers = Math.max(aLayers.length, bLayers.length);
    const layerDiffs = [];
    for (let i = 0; i < maxLayers; i++) {
      const aL = aLayers[i];
      const bL = bLayers[i];
      if (!aL && bL) {
        layerDiffs.push({ type: "added", idx: i, blocks: bL.blocks });
      } else if (aL && !bL) {
        layerDiffs.push({ type: "removed", idx: i, blocks: aL.blocks });
      } else {
        // Compare blocks
        const aBlocks = aL.blocks;
        const bBlocks = bL.blocks;
        const maxBlocks = Math.max(aBlocks.length, bBlocks.length);
        const blockChanges = [];
        let hasChanges = false;
        for (let j = 0; j < maxBlocks; j++) {
          const aB = aBlocks[j];
          const bB = bBlocks[j];
          if (!aB && bB) {
            blockChanges.push({ type: "added", block: bB });
            hasChanges = true;
          } else if (aB && !bB) {
            blockChanges.push({ type: "removed", block: aB });
            hasChanges = true;
          } else {
            // Compare primitives
            const aPrims = aB.primitives.map(p => `${p.type}:${JSON.stringify(p.cfg)}`).join("|");
            const bPrims = bB.primitives.map(p => `${p.type}:${JSON.stringify(p.cfg)}`).join("|");
            if (aPrims !== bPrims || aB.name !== bB.name) {
              blockChanges.push({ type: "changed", a: aB, b: bB });
              hasChanges = true;
            } else {
              blockChanges.push({ type: "same", block: aB });
            }
          }
        }
        layerDiffs.push({ type: hasChanges ? "changed" : "same", idx: i, blockChanges });
      }
    }

    // Config diffs
    const cfgDiffs = [];
    for (const key of Object.keys(aCfg)) {
      if (aCfg[key] !== bCfg[key]) {
        cfgDiffs.push({ key, a: aCfg[key], b: bCfg[key] });
      }
    }

    // Primitive count diffs
    const aPrims = flattenPrims(aLayers);
    const bPrims = flattenPrims(bLayers);
    const aTypeCount = {};
    const bTypeCount = {};
    aPrims.forEach(p => { aTypeCount[p.prim.type] = (aTypeCount[p.prim.type] || 0) + 1; });
    bPrims.forEach(p => { bTypeCount[p.prim.type] = (bTypeCount[p.prim.type] || 0) + 1; });

    return { aParams, bParams, layerDiffs, cfgDiffs, aTypeCount, bTypeCount, aLayerCount: aLayers.length, bLayerCount: bLayers.length };
  }, [compareData, currentLayers, currentCfg]);

  const colorFor = (type) => type === "added" ? "#22c55e" : type === "removed" ? "#ef4444" : type === "changed" ? "#f59e0b" : "#2a3f55";

  return (
    <div style={{ padding: "0 14px 14px" }}>
      {!compareData ? (
        <div style={{ ...S.panel, padding: 16, textAlign: "center" }}>
          <div style={{ ...S.label, marginBottom: 10, color: "#06b6d4" }}>Load a second architecture to compare</div>
          <div style={{ display: "flex", gap: 6, justifyContent: "center" }}>
            <button onClick={() => fileRef.current?.click()} style={{
              background: "#172035", color: "#c8d6e5", border: "1px solid #22354a", borderRadius: 4,
              padding: "8px 16px", fontSize: 9, fontFamily: font, fontWeight: 600, cursor: "pointer",
              textTransform: "uppercase", letterSpacing: "0.08em",
            }}>Load from File</button>
            <button onClick={loadFromClipboard} style={{
              background: "#172035", color: "#c8d6e5", border: "1px solid #22354a", borderRadius: 4,
              padding: "8px 16px", fontSize: 9, fontFamily: font, fontWeight: 600, cursor: "pointer",
              textTransform: "uppercase", letterSpacing: "0.08em",
            }}>Paste from Clipboard</button>
          </div>
          <input ref={fileRef} type="file" accept=".json" onChange={loadFromFile} style={{ display: "none" }} />
        </div>
      ) : (
        <>
          {/* Summary */}
          <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <div style={{ ...S.label, color: "#06b6d4" }}>Comparison Summary</div>
              <button onClick={() => setCompareData(null)} style={{
                background: "none", border: "1px solid #172035", borderRadius: 3, color: "#6b8299",
                cursor: "pointer", fontSize: 7, fontFamily: font, padding: "2px 6px",
              }}>Clear</button>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 8, alignItems: "center" }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 7, color: "#3e5775", textTransform: "uppercase" }}>Current</div>
                <div style={{ fontSize: 12, color: "#22c55e", fontWeight: 700 }}>{fmt(diff.aParams)}</div>
                <div style={{ fontSize: 8, color: "#3e5775" }}>{diff.aLayerCount}L</div>
              </div>
              <div style={{ fontSize: 10, color: "#3e5775" }}>vs</div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 7, color: "#3e5775", textTransform: "uppercase" }}>{compareData.meta.name}</div>
                <div style={{ fontSize: 12, color: "#06b6d4", fontWeight: 700 }}>{fmt(diff.bParams)}</div>
                <div style={{ fontSize: 8, color: "#3e5775" }}>{diff.bLayerCount}L</div>
              </div>
            </div>
            <div style={{ textAlign: "center", marginTop: 6, fontSize: 9, fontWeight: 600, color: diff.aParams > diff.bParams ? "#ef4444" : "#22c55e" }}>
              {diff.aParams !== diff.bParams ? `${diff.aParams > diff.bParams ? "+" : ""}${fmt(diff.aParams - diff.bParams)} params` : "Same parameter count"}
            </div>
          </div>

          {/* Config diffs */}
          {diff.cfgDiffs.length > 0 && (
            <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
              <div style={{ ...S.label, marginBottom: 6, color: "#f59e0b" }}>Config Differences</div>
              {diff.cfgDiffs.map(d => (
                <div key={d.key} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", fontSize: 9 }}>
                  <span style={{ color: "#6b8299" }}>{d.key}</span>
                  <span>
                    <span style={{ color: "#ef4444", textDecoration: "line-through" }}>{d.a}</span>
                    <span style={{ color: "#3e5775", margin: "0 4px" }}>→</span>
                    <span style={{ color: "#22c55e" }}>{d.b}</span>
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Layer-by-layer diff */}
          <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
            <div style={{ ...S.label, marginBottom: 6, color: "#a855f7" }}>Layer Diff</div>
            {diff.layerDiffs.map((ld, i) => (
              <div key={i} style={{
                padding: "3px 6px", marginBottom: 2, borderRadius: 3,
                borderLeft: `2px solid ${colorFor(ld.type)}`,
                background: ld.type === "same" ? "transparent" : `${colorFor(ld.type)}08`,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 9, color: colorFor(ld.type), fontWeight: ld.type !== "same" ? 600 : 400 }}>
                    Layer {i + 1}
                    {ld.type === "added" && " (new)"}
                    {ld.type === "removed" && " (removed)"}
                  </span>
                  {ld.blockChanges && (
                    <span style={{ fontSize: 7, color: "#3e5775" }}>
                      {ld.blockChanges.filter(b => b.type !== "same").length > 0
                        ? `${ld.blockChanges.filter(b => b.type !== "same").length} block changes`
                        : "identical"}
                    </span>
                  )}
                </div>
                {ld.type === "changed" && ld.blockChanges && (
                  <div style={{ marginTop: 2, paddingLeft: 8 }}>
                    {ld.blockChanges.filter(b => b.type !== "same").map((bc, j) => (
                      <div key={j} style={{ fontSize: 8, color: colorFor(bc.type), padding: "1px 0" }}>
                        {bc.type === "added" && `+ ${bc.block.name} (${bc.block.primitives.length} ops)`}
                        {bc.type === "removed" && `- ${bc.block.name} (${bc.block.primitives.length} ops)`}
                        {bc.type === "changed" && `~ ${bc.a.name} → ${bc.b.name} (${bc.a.primitives.length} → ${bc.b.primitives.length} ops)`}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Primitive type comparison */}
          <div style={{ ...S.panel, padding: "8px 10px" }}>
            <div style={{ ...S.label, marginBottom: 6, color: "#ec4899" }}>Primitive Counts</div>
            {(() => {
              const allTypes = new Set([...Object.keys(diff.aTypeCount), ...Object.keys(diff.bTypeCount)]);
              return [...allTypes].sort().map(type => {
                const reg = PRIM[type];
                if (!reg) return null;
                const a = diff.aTypeCount[type] || 0;
                const b = diff.bTypeCount[type] || 0;
                const delta = a - b;
                return (
                  <div key={type} style={{ display: "flex", justifyContent: "space-between", padding: "2px 0", fontSize: 9 }}>
                    <span style={{ color: reg.color }}>{reg.icon} {reg.name}</span>
                    <span>
                      <span style={{ color: "#6b8299" }}>{a}</span>
                      <span style={{ color: "#3e5775", margin: "0 3px" }}>vs</span>
                      <span style={{ color: "#6b8299" }}>{b}</span>
                      {delta !== 0 && <span style={{ color: delta > 0 ? "#22c55e" : "#ef4444", marginLeft: 4, fontSize: 8 }}>({delta > 0 ? "+" : ""}{delta})</span>}
                    </span>
                  </div>
                );
              });
            })()}
          </div>
        </>
      )}
    </div>
  );
}
