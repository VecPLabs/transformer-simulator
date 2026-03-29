import { useState, useMemo } from "react";
import { PRIM } from "../data/primitives";
import { resolve } from "../data/dimensions";
import { S, font } from "./styles";

function indent(s, n = 1) {
  const pad = "    ".repeat(n);
  return s.split("\n").map(l => pad + l).join("\n");
}

function sanitize(name) {
  return name.replace(/[^a-zA-Z0-9_]/g, "_").replace(/^[0-9]/, "_$&");
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

  // Generate a block class
  const blockClasses = new Map();
  const blockClassNames = [];

  if (layers.length > 0) {
    // Deduplicate: blocks with same primitive signature share a class
    layers[0].blocks.forEach((block, bIdx) => {
      const sig = block.primitives.map(p => `${p.type}:${JSON.stringify(p.cfg)}`).join("|");
      if (blockClasses.has(sig)) {
        blockClassNames.push(blockClasses.get(sig).className);
        return;
      }

      const className = sanitize(block.name || `Block${bIdx}`);
      blockClasses.set(sig, { className, block });
      blockClassNames.push(className);

      lines.push(`class ${className}(nn.Module):`);
      lines.push(`    def __init__(self, config: ModelConfig):`);
      lines.push(`        super().__init__()`);

      // Init lines
      let fieldIdx = 0;
      block.primitives.forEach(p => {
        const reg = PRIM[p.type];
        if (!reg) return;
        const name = sanitize(p.label || `op_${fieldIdx}`);
        const inD = resolve(p.cfg.inDim || "hiddenDim", globalCfg);
        const outD = resolve(p.cfg.outDim || "hiddenDim", globalCfg);

        switch (p.type) {
          case "linear":
            lines.push(`        self.${name} = nn.Linear(${inD}, ${outD}, bias=False)`);
            fieldIdx++;
            break;
          case "linear_bias":
            lines.push(`        self.${name} = nn.Linear(${inD}, ${outD}, bias=True)`);
            fieldIdx++;
            break;
          case "rmsnorm":
            lines.push(`        self.${name} = RMSNorm(${resolve(p.cfg.dim || "hiddenDim", globalCfg)})`);
            fieldIdx++;
            break;
          case "layernorm":
            lines.push(`        self.${name} = nn.LayerNorm(${resolve(p.cfg.dim || "hiddenDim", globalCfg)})`);
            fieldIdx++;
            break;
          case "dropout":
            lines.push(`        self.${name} = nn.Dropout(${p.cfg.rate || 0.1})`);
            fieldIdx++;
            break;
          default:
            // No learnable params
            break;
        }
      });

      lines.push(``);
      lines.push(`    def forward(self, x, residual=None):`);

      // Forward lines
      let hasGate = false;
      block.primitives.forEach(p => {
        const name = sanitize(p.label || `op_${fieldIdx}`);
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
          case "gate_mul":
            lines.push(`        x = gate * x  # element-wise gating`);
            break;
          case "residual_add":
            lines.push(`        x = x + residual if residual is not None else x`);
            break;
          case "rope":
            lines.push(`        x = apply_rope(x, x.shape[-2])`);
            break;
          case "scale_dot": {
            const h = resolve(p.cfg.numHeads || "heads", globalCfg);
            const d = resolve(p.cfg.headDim || "headDim", globalCfg);
            lines.push(`        # Scaled dot-product attention (${h} heads, dim ${d})`);
            lines.push(`        x = F.scaled_dot_product_attention(q, k, v)  # placeholder — reshape Q/K/V first`);
            break;
          }
          case "dropout":
            lines.push(`        x = self.${name}(x)`);
            break;
          case "router":
            lines.push(`        # MoE routing — top-${resolve(p.cfg.topK || "moeTopK", globalCfg)} of ${resolve(p.cfg.numExperts || "moeExperts", globalCfg)} experts`);
            lines.push(`        x = x  # TODO: implement expert routing`);
            break;
          case "split":
            lines.push(`        residual = x  # branch / save for later`);
            break;
          case "concat":
            lines.push(`        x = torch.cat([x, residual], dim=-1)  # concat`);
            break;
          case "custom_note":
            lines.push(`        # ${p.cfg.text || "annotation"}`);
            break;
          default:
            lines.push(`        # ${p.type}: ${p.label}`);
        }
      });

      lines.push(`        return x`);
      lines.push(``);
      lines.push(``);
    });
  }

  // Main model class
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
    lines.push(`        for layer_blocks in self.layers:`);
    lines.push(`            residual = x`);
    lines.push(`            for block in layer_blocks:`);
    lines.push(`                x = block(x, residual=residual)`);
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
    <div style={{ ...S.panel, padding: 0, background: "#080c16", border: "1px solid #22c55e22" }}>
      <div style={{
        padding: "8px 10px", borderBottom: "1px solid #172035",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <div style={{ ...S.label, color: "#22c55e" }}>PyTorch Export</div>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={copyCode} style={{
            background: "#172035", color: "#c8d6e5", border: "1px solid #22354a",
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
        color: "#a0b4c8", fontFamily: font, overflow: "auto", maxHeight: 400,
        whiteSpace: "pre", tabSize: 4,
      }}>{code}</pre>
    </div>
  );
}
