export const ARCH_PRESETS = {
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
