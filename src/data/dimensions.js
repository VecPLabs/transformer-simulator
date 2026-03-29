export const DIM_OPTIONS = [
  { value: "hiddenDim", label: "Hidden Dim" },
  { value: "ffnDim", label: "FFN Dim" },
  { value: "heads", label: "Num Heads" },
  { value: "kvHeads", label: "KV Heads" },
  { value: "headDim", label: "Head Dim" },
  { value: "vocabSize", label: "Vocab Size" },
  { value: "moeExperts", label: "MoE Experts" },
  { value: "moeTopK", label: "MoE Top-K" },
];

export function resolve(ref, globalCfg) {
  if (typeof ref === "number") return ref;
  if (ref === "headDim") return Math.floor(globalCfg.hiddenDim / globalCfg.heads);
  if (ref === "kvDim") return globalCfg.kvHeads * Math.floor(globalCfg.hiddenDim / globalCfg.heads);
  return globalCfg[ref] || 0;
}
