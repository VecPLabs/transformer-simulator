import { useMemo } from "react";
import { PRIM } from "../data/primitives";
import { resolve } from "../data/dimensions";
import { S, font } from "./styles";

function sanitize(name) {
  return name.replace(/[^a-zA-Z0-9_]/g, "_").replace(/^[0-9]/, "_$&");
}

// ─── Dataflow Analysis ────────────────────────────────────────────────
// Detect structural patterns in a block's primitives to emit correct code.

function analyzeBlock(primitives) {
  // Find scale_dot index
  const scaleDotIdx = primitives.findIndex(p => p.type === "scale_dot");
  // Find gate_mul index
  const gateMulIdx = primitives.findIndex(p => p.type === "gate_mul");

  // Detect attention fan-out: 3+ linear projections before a scale_dot
  let attnPattern = null;
  if (scaleDotIdx >= 0) {
    const linearsBeforeAttn = [];
    for (let i = 0; i < scaleDotIdx; i++) {
      if (primitives[i].type === "linear" || primitives[i].type === "linear_bias") {
        linearsBeforeAttn.push(i);
      }
    }
    if (linearsBeforeAttn.length >= 3) {
      attnPattern = {
        qIdx: linearsBeforeAttn[0],
        kIdx: linearsBeforeAttn[1],
        vIdx: linearsBeforeAttn[2],
        scaleDotIdx,
        // Output projection is typically the linear right after scale_dot
        outIdx: primitives.findIndex((p, i) =>
          i > scaleDotIdx && (p.type === "linear" || p.type === "linear_bias")),
      };
    }
  }

  // Detect SwiGLU pattern: two linears before gate_mul, with activation on the first path
  let swigluPattern = null;
  if (gateMulIdx >= 0) {
    const linearsBeforeGate = [];
    for (let i = 0; i < gateMulIdx; i++) {
      if (primitives[i].type === "linear" || primitives[i].type === "linear_bias") {
        linearsBeforeGate.push(i);
      }
    }
    if (linearsBeforeGate.length >= 2) {
      // Find activation between first linear and gate_mul (belongs to gate path)
      const gateLinearIdx = linearsBeforeGate[linearsBeforeGate.length - 2];
      const upLinearIdx = linearsBeforeGate[linearsBeforeGate.length - 1];
      let activationIdx = -1;
      for (let i = gateLinearIdx + 1; i < gateMulIdx; i++) {
        const t = primitives[i].type;
        if (t === "silu" || t === "gelu" || t === "relu") {
          activationIdx = i;
          break;
        }
      }
      swigluPattern = { gateLinearIdx, activationIdx, upLinearIdx, gateMulIdx };
    }
  }

  return { attnPattern, swigluPattern };
}

// ─── Code Generator ───────────────────────────────────────────────────

function generateBlockForward(primitives, globalCfg, analysis) {
  const lines = [];
  const { attnPattern, swigluPattern } = analysis;

  // Track which primitives are handled by pattern emission
  const handled = new Set();

  // Pre-emit attention pattern
  if (attnPattern) {
    handled.add(attnPattern.qIdx);
    handled.add(attnPattern.kIdx);
    handled.add(attnPattern.vIdx);
    handled.add(attnPattern.scaleDotIdx);
    // Also handle any rope between K/V and scale_dot
    for (let i = attnPattern.vIdx + 1; i < attnPattern.scaleDotIdx; i++) {
      if (primitives[i].type === "rope") handled.add(i);
    }
    if (attnPattern.outIdx >= 0) handled.add(attnPattern.outIdx);
  }

  // Pre-emit SwiGLU pattern
  if (swigluPattern) {
    handled.add(swigluPattern.gateLinearIdx);
    handled.add(swigluPattern.upLinearIdx);
    handled.add(swigluPattern.gateMulIdx);
    if (swigluPattern.activationIdx >= 0) handled.add(swigluPattern.activationIdx);
  }

  // Emit primitives in order, using pattern-aware emission for detected patterns
  for (let i = 0; i < primitives.length; i++) {
    const p = primitives[i];
    const name = sanitize(p.label || `op_${i}`);

    // --- Attention pattern emission ---
    if (attnPattern && i === attnPattern.qIdx) {
      const qName = sanitize(primitives[attnPattern.qIdx].label || "W_Q");
      const kName = sanitize(primitives[attnPattern.kIdx].label || "W_K");
      const vName = sanitize(primitives[attnPattern.vIdx].label || "W_V");
      const h = resolve(primitives[attnPattern.scaleDotIdx].cfg.numHeads || "heads", globalCfg);
      const d = resolve(primitives[attnPattern.scaleDotIdx].cfg.headDim || "headDim", globalCfg);
      const hasRope = [...handled].some(idx => idx > attnPattern.vIdx && idx < attnPattern.scaleDotIdx && primitives[idx].type === "rope");

      lines.push(`        h = x`);
      lines.push(`        bsz, seq_len, _ = h.shape`);
      lines.push(`        q = self.${qName}(h).view(bsz, seq_len, ${h}, ${d}).transpose(1, 2)`);
      lines.push(`        k = self.${kName}(h).view(bsz, seq_len, -1, ${d}).transpose(1, 2)`);
      lines.push(`        v = self.${vName}(h).view(bsz, seq_len, -1, ${d}).transpose(1, 2)`);
      if (hasRope) {
        lines.push(`        q = apply_rope(q, seq_len)`);
        lines.push(`        k = apply_rope(k, seq_len)`);
      }
      lines.push(`        x = F.scaled_dot_product_attention(q, k, v)`);
      lines.push(`        x = x.transpose(1, 2).contiguous().view(bsz, seq_len, -1)`);
      if (attnPattern.outIdx >= 0) {
        const outName = sanitize(primitives[attnPattern.outIdx].label || "W_O");
        lines.push(`        x = self.${outName}(x)`);
      }
      continue;
    }

    // --- SwiGLU pattern emission ---
    if (swigluPattern && i === swigluPattern.gateLinearIdx) {
      const gateName = sanitize(primitives[swigluPattern.gateLinearIdx].label || "W_gate");
      const upName = sanitize(primitives[swigluPattern.upLinearIdx].label || "W_up");
      const actType = swigluPattern.activationIdx >= 0 ? primitives[swigluPattern.activationIdx].type : "silu";
      const actFn = actType === "gelu" ? "F.gelu" : actType === "relu" ? "F.relu" : "F.silu";

      lines.push(`        h = x`);
      lines.push(`        gate = ${actFn}(self.${gateName}(h))`);
      lines.push(`        up = self.${upName}(h)`);
      lines.push(`        x = gate * up`);
      continue;
    }

    // Skip handled primitives
    if (handled.has(i)) continue;

    // --- Default sequential emission ---
    switch (p.type) {
      case "linear":
      case "linear_bias":
        lines.push(`        x = self.${name}(x)`);
        break;
      case "rmsnorm":
      case "layernorm":
        lines.push(`        x = self.${name}(x)`);
        break;
      case "silu":
        lines.push(`        x = F.silu(x)`);
        break;
      case "gelu":
        lines.push(`        x = F.gelu(x)`);
        break;
      case "relu":
        lines.push(`        x = F.relu(x)`);
        break;
      case "softmax":
        lines.push(`        x = F.softmax(x, dim=-1)`);
        break;
      case "residual_add":
        lines.push(`        x = x + residual`);
        break;
      case "dropout":
        lines.push(`        x = self.${name}(x)`);
        break;
      case "rope":
        lines.push(`        x = apply_rope(x, x.shape[-2])`);
        break;
      case "router":
        lines.push(`        # MoE routing — top-${resolve(p.cfg.topK || "moeTopK", globalCfg)} of ${resolve(p.cfg.numExperts || "moeExperts", globalCfg)} experts`);
        lines.push(`        x = x  # TODO: implement expert routing`);
        break;
      case "split":
        lines.push(`        residual = x`);
        break;
      case "concat":
        lines.push(`        x = torch.cat([x, residual], dim=-1)`);
        break;
      case "custom_note":
        lines.push(`        # ${p.cfg.text || "annotation"}`);
        break;
      case "gate_mul":
        lines.push(`        x = gate * x  # element-wise gating`);
        break;
      case "scale_dot": {
        const h = resolve(p.cfg.numHeads || "heads", globalCfg);
        const d = resolve(p.cfg.headDim || "headDim", globalCfg);
        lines.push(`        # Scaled dot-product attention (${h} heads, dim ${d})`);
        lines.push(`        x = F.scaled_dot_product_attention(q, k, v)`);
        break;
      }
      default:
        lines.push(`        # ${p.type}: ${p.label}`);
    }
  }

  return lines;
}

function generatePyTorch(layers, globalCfg) {
  const hiddenDim = globalCfg.hiddenDim;
  const ffnDim = globalCfg.ffnDim;
  const heads = globalCfg.heads;
  const kvHeads = globalCfg.kvHeads;
  const headDim = Math.floor(hiddenDim / heads);
  const vocabSize = globalCfg.vocabSize;
  const seqLen = globalCfg.seqLen;

  const lines = [];
  lines.push(`import torch`);
  lines.push(`import torch.nn as nn`);
  lines.push(`import torch.nn.functional as F`);
  lines.push(`import math`);
  lines.push(`from dataclasses import dataclass`);
  lines.push(``);
  lines.push(``);
  lines.push(`@dataclass`);
  lines.push(`class ModelConfig:`);
  lines.push(`    hidden_dim: int = ${hiddenDim}`);
  lines.push(`    ffn_dim: int = ${ffnDim}`);
  lines.push(`    num_heads: int = ${heads}`);
  lines.push(`    num_kv_heads: int = ${kvHeads}`);
  lines.push(`    head_dim: int = ${headDim}`);
  lines.push(`    vocab_size: int = ${vocabSize}`);
  lines.push(`    max_seq_len: int = ${seqLen}`);
  lines.push(`    num_layers: int = ${layers.length}`);
  lines.push(``);
  lines.push(``);

  // RMSNorm helper
  lines.push(`class RMSNorm(nn.Module):`);
  lines.push(`    def __init__(self, dim, eps=1e-6):`);
  lines.push(`        super().__init__()`);
  lines.push(`        self.weight = nn.Parameter(torch.ones(dim))`);
  lines.push(`        self.eps = eps`);
  lines.push(``);
  lines.push(`    def forward(self, x):`);
  lines.push(`        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight`);
  lines.push(``);
  lines.push(``);

  // RoPE helper
  lines.push(`def apply_rope(x, seq_len):`);
  lines.push(`    """Simplified RoPE — replace with full implementation for production."""`);
  lines.push(`    d = x.shape[-1]`);
  lines.push(`    pos = torch.arange(seq_len, device=x.device).unsqueeze(1)`);
  lines.push(`    dim = torch.arange(0, d, 2, device=x.device).float()`);
  lines.push(`    freqs = pos / (10000.0 ** (dim / d))`);
  lines.push(`    cos_f, sin_f = freqs.cos(), freqs.sin()`);
  lines.push(`    x1, x2 = x[..., ::2], x[..., 1::2]`);
  lines.push(`    return torch.stack([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1).flatten(-2)`);
  lines.push(``);
  lines.push(``);

  // Generate block classes
  const blockClasses = new Map();
  const blockClassNames = [];

  if (layers.length > 0) {
    layers[0].blocks.forEach((block, bIdx) => {
      const sig = block.primitives.map(p => `${p.type}:${JSON.stringify(p.cfg)}`).join("|");
      if (blockClasses.has(sig)) {
        blockClassNames.push(blockClasses.get(sig).className);
        return;
      }

      const className = sanitize(block.name || `Block${bIdx}`);
      blockClasses.set(sig, { className, block });
      blockClassNames.push(className);

      const analysis = analyzeBlock(block.primitives);

      lines.push(`class ${className}(nn.Module):`);
      lines.push(`    def __init__(self, config: ModelConfig):`);
      lines.push(`        super().__init__()`);

      // Init: emit nn.Module attributes for primitives with learnable params
      block.primitives.forEach((p, pIdx) => {
        const name = sanitize(p.label || `op_${pIdx}`);
        const inD = resolve(p.cfg.inDim || "hiddenDim", globalCfg);
        const outD = resolve(p.cfg.outDim || "hiddenDim", globalCfg);

        switch (p.type) {
          case "linear":
            lines.push(`        self.${name} = nn.Linear(${inD}, ${outD}, bias=False)`);
            break;
          case "linear_bias":
            lines.push(`        self.${name} = nn.Linear(${inD}, ${outD}, bias=True)`);
            break;
          case "rmsnorm":
            lines.push(`        self.${name} = RMSNorm(${resolve(p.cfg.dim || "hiddenDim", globalCfg)})`);
            break;
          case "layernorm":
            lines.push(`        self.${name} = nn.LayerNorm(${resolve(p.cfg.dim || "hiddenDim", globalCfg)})`);
            break;
          case "dropout":
            lines.push(`        self.${name} = nn.Dropout(${p.cfg.rate || 0.1})`);
            break;
        }
      });

      lines.push(``);
      lines.push(`    def forward(self, x, residual=None):`);

      // Forward: pattern-aware emission
      const fwdLines = generateBlockForward(block.primitives, globalCfg, analysis);
      fwdLines.forEach(l => lines.push(l));

      lines.push(`        return x`);
      lines.push(``);
      lines.push(``);
    });
  }

  // Main model class
  const firstLayerTopology = layers.length > 0 ? (layers[0].topology || "sequential") : "sequential";
  const hasParallel = layers.some(l => l.topology === "parallel" || l.topology === "parallel_then_sequential");

  lines.push(`class TransformerModel(nn.Module):`);
  lines.push(`    def __init__(self, config: ModelConfig):`);
  lines.push(`        super().__init__()`);
  lines.push(`        self.config = config`);
  lines.push(`        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)`);

  if (blockClassNames.length > 0) {
    lines.push(`        self.layers = nn.ModuleList([`);
    lines.push(`            nn.ModuleList([`);
    blockClassNames.forEach(cn => {
      lines.push(`                ${cn}(config),`);
    });
    lines.push(`            ]) for _ in range(config.num_layers)`);
    lines.push(`        ])`);
  }
  lines.push(`        self.norm = RMSNorm(config.hidden_dim)`);
  lines.push(`        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)`);
  lines.push(``);
  lines.push(`    def forward(self, input_ids):`);
  lines.push(`        x = self.embed(input_ids)`);

  if (blockClassNames.length > 0) {
    if (hasParallel) {
      // Layer topology is per-layer; for code gen we use the first layer's topology as representative
      const topo = firstLayerTopology;
      if (topo === "parallel") {
        lines.push(`        for layer_blocks in self.layers:`);
        lines.push(`            residual = x`);
        lines.push(`            branch_sum = sum(block(x) for block in layer_blocks)`);
        lines.push(`            x = residual + branch_sum`);
      } else if (topo === "parallel_then_sequential") {
        // Find the split point from the first layer
        const parallelCount = layers[0].parallelCount || layers[0].blocks.length - 1;
        lines.push(`        for layer_blocks in self.layers:`);
        lines.push(`            residual = x`);
        lines.push(`            # Parallel branches`);
        lines.push(`            branch_sum = sum(layer_blocks[i](x) for i in range(${parallelCount}))`);
        lines.push(`            x = residual + branch_sum`);
        lines.push(`            # Sequential blocks`);
        lines.push(`            for block in layer_blocks[${parallelCount}:]:`);
        lines.push(`                residual = x`);
        lines.push(`                x = block(x, residual=residual)`);
      }
    } else {
      lines.push(`        for layer_blocks in self.layers:`);
      lines.push(`            residual = x`);
      lines.push(`            for block in layer_blocks:`);
      lines.push(`                x = block(x, residual=residual)`);
    }
  }

  lines.push(`        x = self.norm(x)`);
  lines.push(`        return self.lm_head(x)`);
  lines.push(``);
  lines.push(``);
  lines.push(`if __name__ == "__main__":`);
  lines.push(`    config = ModelConfig()`);
  lines.push(`    model = TransformerModel(config)`);
  lines.push(`    total = sum(p.numel() for p in model.parameters())`);
  lines.push(`    print(f"Parameters: {total:,}")`);
  lines.push(`    print(f"  Embedding: {model.embed.weight.numel():,}")`);
  lines.push(`    print(f"  LM Head:   {model.lm_head.weight.numel():,}")`);
  lines.push(`    if hasattr(model, 'layers'):`);
  lines.push(`        layer_p = sum(p.numel() for l in model.layers for p in l.parameters())`);
  lines.push(`        print(f"  Layers:    {layer_p:,} ({layer_p // config.num_layers:,}/layer)")`);

  return lines.join("\n");
}

export default function PyTorchExport({ layers, globalCfg, archName, showToast }) {
  const code = useMemo(() => generatePyTorch(layers, globalCfg), [layers, globalCfg]);

  const copyCode = async () => {
    try {
      await navigator.clipboard.writeText(code);
      showToast("PyTorch code copied!");
    } catch {
      showToast("Failed to copy", "error");
    }
  };

  const downloadFile = () => {
    const blob = new Blob([code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeName = (archName || "model").replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
    a.download = `${safeName}.py`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Downloaded .py file");
  };

  return (
    <div style={{ ...S.panel, padding: 0, background: "var(--panel-bg-deep)", border: "1px solid var(--accent-green, #22c55e)22" }}>
      <div style={{
        padding: "8px 10px", borderBottom: "1px solid var(--border)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <div style={{ ...S.label, color: "var(--accent-green)" }}>PyTorch Export</div>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={copyCode} style={{
            background: "var(--border)", color: "var(--text)", border: "1px solid var(--muted-faint)",
            borderRadius: 3, padding: "3px 8px", fontSize: 8, fontFamily: font,
            fontWeight: 600, cursor: "pointer",
          }}>Copy</button>
          <button onClick={downloadFile} style={{
            background: "linear-gradient(135deg, #22c55e, #16a34a)", color: "#000",
            border: "none", borderRadius: 3, padding: "3px 8px", fontSize: 8,
            fontFamily: font, fontWeight: 700, cursor: "pointer",
          }}>Download .py</button>
        </div>
      </div>
      <pre style={{
        padding: "8px 10px", margin: 0, fontSize: 9, lineHeight: 1.5,
        color: "var(--text-secondary)", fontFamily: font, overflow: "auto", maxHeight: 400,
        whiteSpace: "pre", tabSize: 4,
      }}>{code}</pre>
    </div>
  );
}
