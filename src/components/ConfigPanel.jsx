import { S, font } from "./styles";

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

export default function ConfigPanel({ globalCfg, updateCfg }) {
  return (
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
  );
}
