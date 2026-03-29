import { resolve } from "./dimensions";

// outShape(cfg, g, inDim) → number: output last-dim given input last-dim
// flops(cfg, g, seqLen) → number: FLOPs per token (simplified)

export const PRIM = {
  linear: {
    name: "Linear Projection", icon: "─", color: "#ec4899", category: "projection",
    desc: "Wx (learnable matrix multiply)",
    info: "Multiplies the input by a learned weight matrix W. This is the fundamental building block of transformers — used for Q/K/V projections, output projections, and FFN layers. No bias term.",
    defaultCfg: { inDim: "hiddenDim", outDim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.inDim, g) * resolve(cfg.outDim, g),
    shape: (cfg, g) => `[${resolve(cfg.inDim, g)}, ${resolve(cfg.outDim, g)}]`,
    outShape: (cfg, g) => resolve(cfg.outDim, g),
    expectedIn: (cfg, g) => resolve(cfg.inDim, g),
    flops: (cfg, g) => 2 * resolve(cfg.inDim, g) * resolve(cfg.outDim, g),
    configFields: [
      { key: "inDim", label: "Input Dim", type: "dimSelect" },
      { key: "outDim", label: "Output Dim", type: "dimSelect" },
    ],
  },
  linear_bias: {
    name: "Linear + Bias", icon: "━", color: "#f472b6", category: "projection",
    desc: "Wx + b (projection with bias)",
    info: "Like Linear Projection but adds a learned bias vector b after the matrix multiply. Used in GPT-2 style models. Modern architectures (Llama, Mistral) often omit the bias for efficiency.",
    defaultCfg: { inDim: "hiddenDim", outDim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.inDim, g) * resolve(cfg.outDim, g) + resolve(cfg.outDim, g),
    shape: (cfg, g) => `[${resolve(cfg.inDim, g)}, ${resolve(cfg.outDim, g)}] + b`,
    outShape: (cfg, g) => resolve(cfg.outDim, g),
    expectedIn: (cfg, g) => resolve(cfg.inDim, g),
    flops: (cfg, g) => 2 * resolve(cfg.inDim, g) * resolve(cfg.outDim, g) + resolve(cfg.outDim, g),
    configFields: [
      { key: "inDim", label: "Input Dim", type: "dimSelect" },
      { key: "outDim", label: "Output Dim", type: "dimSelect" },
    ],
  },
  silu: {
    name: "SiLU / Swish", icon: "∿", color: "#facc15", category: "activation",
    desc: "x · σ(x)",
    info: "Sigmoid Linear Unit — multiplies each value by its own sigmoid. Smooth, non-monotonic activation used in Llama, PaLM, and most modern FFN blocks (as part of SwiGLU gating).",
    defaultCfg: {}, params: () => 0, shape: () => "SiLU(x)",
    outShape: (cfg, g, inDim) => inDim,
    flops: (cfg, g, inDim) => 5 * (inDim || 0),
    configFields: [],
  },
  gelu: {
    name: "GELU", icon: "≈", color: "#fbbf24", category: "activation",
    desc: "Gaussian Error Linear Unit",
    info: "Approximates x multiplied by the probability that x is positive under a Gaussian. Used in BERT, GPT-2, and GPT-3. Smoother than ReLU, similar in spirit to SiLU.",
    defaultCfg: {}, params: () => 0, shape: () => "GELU(x)",
    outShape: (cfg, g, inDim) => inDim,
    flops: (cfg, g, inDim) => 8 * (inDim || 0),
    configFields: [],
  },
  relu: {
    name: "ReLU", icon: "∠", color: "#f59e0b", category: "activation",
    desc: "max(0, x)",
    info: "Rectified Linear Unit — zeros out negative values, passes positives unchanged. Simple and fast but can cause 'dead neurons'. Largely replaced by GELU/SiLU in modern transformers.",
    defaultCfg: {}, params: () => 0, shape: () => "ReLU(x)",
    outShape: (cfg, g, inDim) => inDim,
    flops: (cfg, g, inDim) => inDim || 0,
    configFields: [],
  },
  softmax: {
    name: "Softmax", icon: "σ", color: "#fde68a", category: "activation",
    desc: "Normalized exponential",
    info: "Converts a vector of raw scores into a probability distribution (all values 0-1, summing to 1). Used in attention to determine how much each token attends to every other token.",
    defaultCfg: {}, params: () => 0, shape: () => "softmax(x)",
    outShape: (cfg, g, inDim) => inDim,
    flops: (cfg, g, inDim) => 5 * (inDim || 0),
    configFields: [],
  },
  gate_mul: {
    name: "Gate Multiply", icon: "⊙", color: "#a78bfa", category: "operation",
    desc: "Element-wise a ⊙ b (gating)",
    info: "Element-wise multiplication of two tensors. In SwiGLU FFNs, one path goes through an activation (the 'gate') and is multiplied with another linear path. This lets the network learn which features to pass through.",
    defaultCfg: {}, params: () => 0, shape: () => "a ⊙ b",
    outShape: (cfg, g, inDim) => inDim,
    flops: (cfg, g, inDim) => inDim || 0,
    configFields: [],
  },
  residual_add: {
    name: "Residual Add", icon: "⊞", color: "#6366f1", category: "operation",
    desc: "x + sublayer(x)",
    info: "Adds the sublayer's output back to its input (skip connection). Critical for training deep networks — gradients flow directly through the addition, preventing vanishing gradients. Every transformer layer uses this.",
    defaultCfg: {}, params: () => 0, shape: () => "x + f(x)",
    outShape: (cfg, g, inDim) => inDim,
    flops: (cfg, g, inDim) => inDim || 0,
    configFields: [],
  },
  scale_dot: {
    name: "Scaled Dot-Product", icon: "⊛", color: "#22d3ee", category: "operation",
    desc: "softmax(QKᵀ/√d)V",
    info: "The core attention mechanism. Computes similarity between queries (Q) and keys (K), scales by √d to prevent extreme values, applies softmax to get attention weights, then uses those weights to mix values (V). This is how tokens 'look at' each other.",
    defaultCfg: { numHeads: "heads", headDim: "headDim" },
    params: () => 0,
    shape: (cfg, g) => `${resolve(cfg.numHeads, g)}h × ${resolve(cfg.headDim, g)}d`,
    outShape: (cfg, g) => resolve(cfg.numHeads, g) * resolve(cfg.headDim, g),
    flops: (cfg, g) => {
      const h = resolve(cfg.numHeads, g), d = resolve(cfg.headDim, g), s = g.seqLen || 1;
      // QK^T: 2*s*h*d per token, softmax: 5*s*h, attn*V: 2*s*h*d per token
      return 2 * s * h * d + 5 * s * h + 2 * s * h * d;
    },
    configFields: [
      { key: "numHeads", label: "Heads", type: "dimSelect" },
      { key: "headDim", label: "Head Dim", type: "dimSelect" },
    ],
  },
  rmsnorm: {
    name: "RMSNorm", icon: "◇", color: "#22c55e", category: "norm",
    desc: "Root Mean Square normalization",
    info: "Normalizes by dividing by the root mean square of the input, then scales by a learned parameter. Simpler and faster than LayerNorm (no mean subtraction or bias). Used in Llama, Mistral, and most modern architectures.",
    defaultCfg: { dim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.dim, g),
    shape: (cfg, g) => `γ [${resolve(cfg.dim, g)}]`,
    outShape: (cfg, g) => resolve(cfg.dim, g),
    expectedIn: (cfg, g) => resolve(cfg.dim, g),
    flops: (cfg, g) => 5 * resolve(cfg.dim, g),
    configFields: [{ key: "dim", label: "Dim", type: "dimSelect" }],
  },
  layernorm: {
    name: "LayerNorm", icon: "◆", color: "#10b981", category: "norm",
    desc: "Layer norm with γ and β",
    info: "Subtracts the mean and divides by standard deviation across features, then applies learned scale (γ) and shift (β). Stabilizes training by keeping activations in a consistent range. Used in GPT-2/3 and BERT.",
    defaultCfg: { dim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.dim, g) * 2,
    shape: (cfg, g) => `γ,β [${resolve(cfg.dim, g)}]`,
    outShape: (cfg, g) => resolve(cfg.dim, g),
    expectedIn: (cfg, g) => resolve(cfg.dim, g),
    flops: (cfg, g) => 8 * resolve(cfg.dim, g),
    configFields: [{ key: "dim", label: "Dim", type: "dimSelect" }],
  },
  dropout: {
    name: "Dropout", icon: "◌", color: "#64748b", category: "regularization",
    desc: "Random zeroing during training",
    info: "Randomly sets a fraction of values to zero during training, forcing the network to not rely on any single feature. Disabled during inference. Helps prevent overfitting, especially in smaller models.",
    defaultCfg: { rate: 0.1 },
    params: () => 0, shape: (cfg) => `p=${cfg.rate}`,
    outShape: (cfg, g, inDim) => inDim,
    flops: () => 0,
    configFields: [{ key: "rate", label: "Rate", type: "number", min: 0, max: 1, step: 0.01 }],
  },
  rope: {
    name: "RoPE", icon: "⟳", color: "#0ea5e9", category: "operation",
    desc: "Rotary Position Embedding (no params)",
    info: "Encodes position by rotating Q and K vectors using sinusoidal frequencies. No learned parameters — position info is baked into the geometry. Enables length generalization and is used in nearly all modern LLMs (Llama, Mistral, etc.).",
    defaultCfg: {}, params: () => 0, shape: () => "rotate(x, θ)",
    outShape: (cfg, g, inDim) => inDim,
    flops: (cfg, g, inDim) => 6 * (inDim || 0),
    configFields: [],
  },
  concat: {
    name: "Concat", icon: "⊕", color: "#8b5cf6", category: "operation",
    desc: "Concatenate along dim",
    info: "Joins two tensors along a dimension (e.g., combining the outputs of multiple attention heads before the output projection). No learnable parameters.",
    defaultCfg: {}, params: () => 0, shape: () => "cat(a, b)",
    outShape: (cfg, g, inDim) => inDim,
    flops: () => 0,
    configFields: [],
  },
  split: {
    name: "Split / Branch", icon: "⑂", color: "#7c3aed", category: "operation",
    desc: "Fork tensor into parallel paths",
    info: "Splits a tensor into multiple parallel paths for independent processing. Used to fork data into Q/K/V streams or to create parallel branches in custom architectures.",
    defaultCfg: {}, params: () => 0, shape: () => "split(x)",
    outShape: (cfg, g, inDim) => inDim,
    flops: () => 0,
    configFields: [],
  },
  router: {
    name: "Router", icon: "⬡", color: "#f59e0b", category: "operation",
    desc: "Top-k expert routing",
    info: "Routes each token to the top-K most relevant experts out of N total. A small learned gate network scores each expert, and only the chosen experts process the token. Enables massive model capacity with lower compute (Mixture of Experts).",
    defaultCfg: { numExperts: "moeExperts", topK: "moeTopK", dim: "hiddenDim" },
    params: (cfg, g) => resolve(cfg.dim, g) * resolve(cfg.numExperts, g),
    shape: (cfg, g) => `top-${resolve(cfg.topK, g)} of ${resolve(cfg.numExperts, g)}`,
    outShape: (cfg, g) => resolve(cfg.dim, g),
    flops: (cfg, g) => 2 * resolve(cfg.dim, g) * resolve(cfg.numExperts, g),
    configFields: [
      { key: "numExperts", label: "Experts", type: "dimSelect" },
      { key: "topK", label: "Top-K", type: "dimSelect" },
    ],
  },
  custom_note: {
    name: "Annotation", icon: "✎", color: "#475569", category: "meta",
    desc: "Non-functional note / label",
    info: "A non-functional label for documentation purposes. Use it to annotate your architecture with notes, mark decision points, or leave reminders. Has no effect on parameter counts.",
    defaultCfg: { text: "custom step" },
    params: () => 0, shape: (cfg) => cfg.text,
    outShape: (cfg, g, inDim) => inDim,
    flops: () => 0,
    configFields: [{ key: "text", label: "Note", type: "text" }],
  },
};

export const PRIM_CATEGORIES = [
  { key: "projection", label: "Projections" },
  { key: "activation", label: "Activations" },
  { key: "norm", label: "Norms" },
  { key: "operation", label: "Operations" },
  { key: "regularization", label: "Regularization" },
  { key: "meta", label: "Meta" },
];

// Propagate shapes through a block's primitives
// Returns array of { inDim, outDim, mismatch } per primitive
export function propagateShapes(primitives, globalCfg) {
  let currentDim = globalCfg.hiddenDim;
  return primitives.map(p => {
    const reg = PRIM[p.type];
    if (!reg) return { inDim: currentDim, outDim: currentDim, mismatch: false };

    const expectedIn = reg.expectedIn ? reg.expectedIn(p.cfg, globalCfg) : null;
    const mismatch = expectedIn !== null && expectedIn !== currentDim;
    const inDim = currentDim;

    if (reg.outShape) {
      currentDim = reg.outShape(p.cfg, globalCfg, currentDim);
    }

    return { inDim, outDim: currentDim, mismatch };
  });
}

// Compute total FLOPs per token for a block's primitives
export function computeBlockFlops(primitives, globalCfg) {
  let currentDim = globalCfg.hiddenDim;
  let total = 0;
  for (const p of primitives) {
    const reg = PRIM[p.type];
    if (!reg) continue;
    if (reg.flops) {
      total += reg.flops(p.cfg, globalCfg, currentDim);
    }
    if (reg.outShape) {
      currentDim = reg.outShape(p.cfg, globalCfg, currentDim);
    }
  }
  return total;
}
