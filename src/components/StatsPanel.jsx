import { PRIM } from "../data/primitives";
import { fmt } from "../utils/format";
import { S } from "./styles";

export default function StatsPanel({ stats, globalCfg }) {
  return (
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

      {/* FLOPs per layer */}
      {stats.totalFlops > 0 && (
        <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
          <div style={{ ...S.label, marginBottom: 6, color: "#06b6d4" }}>FLOPs per Token (per layer)</div>
          <div style={{ maxHeight: 180, overflowY: "auto" }}>
            {stats.perLayer.map((ls, i) => {
              const maxF = Math.max(...stats.perLayer.map(s => s.flops || 0), 1);
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
                  <span style={{ fontSize: 8, color: "#2a3f55", width: 20, textAlign: "right", flexShrink: 0 }}>L{i + 1}</span>
                  <div style={{ flex: 1, height: 8, background: "#080c16", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${(ls.flops || 0) / maxF * 100}%`, background: "linear-gradient(90deg, #06b6d4, #22d3ee)", borderRadius: 2 }} />
                  </div>
                  <span style={{ fontSize: 8, color: "#6b8299", width: 45, textAlign: "right", flexShrink: 0 }}>{fmt(ls.flops || 0)}</span>
                </div>
              );
            })}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4, padding: "3px 0", borderTop: "1px solid #080c16" }}>
            <span style={{ fontSize: 8, color: "#3e5775" }}>Total (incl. fixed)</span>
            <span style={{ fontSize: 9, color: "#06b6d4", fontWeight: 600 }}>{fmt(stats.totalFlops)} FLOPs/token</span>
          </div>
        </div>
      )}

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
            { l: "FLOPs/Param", v: stats.totalP > 0 ? `${(stats.totalFlops / stats.totalP).toFixed(1)}×` : "—" },
          ].map((item, i) => (
            <div key={i} style={{ background: "#080c16", borderRadius: 3, padding: "5px 7px" }}>
              <div style={{ fontSize: 7, color: "#2a3f55", textTransform: "uppercase" }}>{item.l}</div>
              <div style={{ fontSize: 12, color: "#a0b4c8", fontWeight: 600, marginTop: 1 }}>{item.v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
