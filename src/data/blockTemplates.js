export const BLOCK_TEMPLATES = {
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
