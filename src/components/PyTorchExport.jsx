import { useMemo } from "react";
import { PRIM } from "../data/primitives";
import { resolve } from "../data/dimensions";
import { S, font } from "./styles";

function sanitize(name) {
  return name.replace(/[^a-zA-Z0-9_]/g, "_").replace(/^[0-9]/, "_$&");
}

// ─── Dataflow Analysis ────────────────────────────────────────────────

function analyzeBlock(primitives, globalCfg) {
  const scaleDotIdx = primitives.findIndex(p => p.type === "scale_dot");
  const gateMulIdx = primitives.findIndex(p => p.type === "gate_mul");

  // Attention fan-out: 3+ linear projections before a scale_dot
  let attnPattern = null;
  if (scaleDotIdx >= 0) {
    const linearsBeforeAttn = [];
    for (let i = 0; i < scaleDotIdx; i++) {
      if (primitives[i].type === "linear" || primitives[i].type === "linear_bias") {
        linearsBeforeAttn.push(i);
      }
    }
    if (linearsBeforeAttn.length >= 3) {
      // Resolve actual head counts from each projection's output dim
      const qPrim = primitives[linearsBeforeAttn[0]];
      const kPrim = primitives[linearsBeforeAttn[1]];
      const vPrim = primitives[linearsBeforeAttn[2]];
      const sdPrim = primitives[scaleDotIdx];
      const headDim = resolve(sdPrim.cfg.headDim || "headDim", globalCfg);
      const qOutDim = resolve(qPrim.cfg.outDim || "hiddenDim", globalCfg);
      const kOutDim = resolve(kPrim.cfg.outDim || "hiddenDim", globalCfg);
      const qHeads = headDim > 0 ? Math.floor(qOutDim / headDim) : resolve("heads", globalCfg);
      const kvHeads = headDim > 0 ? Math.floor(kOutDim / headDim) : resolve("kvHeads", globalCfg);

      attnPattern = {
        qIdx: linearsBeforeAttn[0],
        kIdx: linearsBeforeAttn[1],
        vIdx: linearsBeforeAttn[2],
        scaleDotIdx,
        outIdx: primitives.findIndex((p, i) =>
          i > scaleDotIdx && (p.type === "linear" || p.type === "linear_bias")),
        qHeads,
        kvHeads,
        headDim,
        needsGQAExpand: kvHeads > 0 && qHeads > kvHeads,
        gqaRepeat: kvHeads > 0 ? Math.floor(qHeads / kvHeads) : 1,
      };
    }
  }

  // SwiGLU: two linears before gate_mul, activation on first path
  let swigluPattern = null;
  if (gateMulIdx >= 0) {
    const linearsBeforeGate = [];
    for (let i = 0; i < gateMulIdx; i++) {
      if (primitives[i].type === "linear" || primitives[i].type === "linear_bias") {
        linearsBeforeGate.push(i);
      }
    }
    if (linearsBeforeGate.length >= 2) {
      const gateLinearIdx = linearsBeforeGate[linearsBeforeGate.length - 2];
      const upLinearIdx = linearsBeforeGate[linearsBeforeGate.length - 1];
      let activationIdx = -1;
      for (let i = gateLinearIdx + 1; i < gateMulIdx; i++) {
        const t = primitives[i].type;
        if (t === "silu" || t === "gelu" || t === "relu") { activationIdx = i; break; }
      }
      swigluPattern = { gateLinearIdx, activationIdx, upLinearIdx, gateMulIdx };
    }
  }

  return { attnPattern, swigluPattern };
}

// ─── Block Forward Code ───────────────────────────────────────────────

function generateBlockForward(primitives, globalCfg, analysis, isParallelBranch) {
  const lines = [];
  const { attnPattern, swigluPattern } = analysis;
  const handled = new Set();

  if (attnPattern) {
    handled.add(attnPattern.qIdx);
    handled.add(attnPattern.kIdx);
    handled.add(attnPattern.vIdx);
    handled.add(attnPattern.scaleDotIdx);
    for (let i = attnPattern.vIdx + 1; i < attnPattern.scaleDotIdx; i++) {
      if (primitives[i].type === "rope") handled.add(i);
    }
    if (attnPattern.outIdx >= 0) handled.add(attnPattern.outIdx);
  }

  if (swigluPattern) {
    handled.add(swigluPattern.gateLinearIdx);
    handled.add(swigluPattern.upLinearIdx);
    handled.add(swigluPattern.gateMulIdx);
    if (swigluPattern.activationIdx >= 0) handled.add(swigluPattern.activationIdx);
  }

  for (let i = 0; i < primitives.length; i++) {
    const p = primitives[i];
    const name = sanitize(p.label || `op_${i}`);

    // --- Attention pattern ---
    if (attnPattern && i === attnPattern.qIdx) {
      const qName = sanitize(primitives[attnPattern.qIdx].label || "W_Q");
      const kName = sanitize(primitives[attnPattern.kIdx].label || "W_K");
      const vName = sanitize(primitives[attnPattern.vIdx].label || "W_V");
      const { qHeads, kvHeads, headDim: hd, needsGQAExpand, gqaRepeat } = attnPattern;
      const hasRope = [...handled].some(idx =>
        idx > attnPattern.vIdx && idx < attnPattern.scaleDotIdx && primitives[idx].type === "rope");

      lines.push(`        h = x`);
      lines.push(`        bsz, seq_len, _ = h.shape`);
      lines.push(`        q = self.${qName}(h).view(bsz, seq_len, ${qHeads}, ${hd}).transpose(1, 2)`);
      lines.push(`        k = self.${kName}(h).view(bsz, seq_len, ${kvHeads}, ${hd}).transpose(1, 2)`);
      lines.push(`        v = self.${vName}(h).view(bsz, seq_len, ${kvHeads}, ${hd}).transpose(1, 2)`);
      if (hasRope) {
        lines.push(`        q = apply_rope(q, seq_len)`);
        lines.push(`        k = apply_rope(k, seq_len)`);
      }
      if (needsGQAExpand) {
        lines.push(`        # GQA: expand ${kvHeads} KV heads to match ${qHeads} Q heads`);
        lines.push(`        k = k.repeat_interleave(${gqaRepeat}, dim=1)`);
        lines.push(`        v = v.repeat_interleave(${gqaRepeat}, dim=1)`);
      }
      lines.push(`        x = F.scaled_dot_product_attention(q, k, v)`);
      lines.push(`        x = x.transpose(1, 2).contiguous().view(bsz, seq_len, -1)`);
      if (attnPattern.outIdx >= 0) {
        const outName = sanitize(primitives[attnPattern.outIdx].label || "W_O");
        lines.push(`        x = self.${outName}(x)`);
      }
      continue;
    }

    // --- SwiGLU pattern ---
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

    if (handled.has(i)) continue;

    // --- Skip residual_add inside parallel branches ---
    if (isParallelBranch && p.type === "residual_add") {
      lines.push(`        # residual add handled at layer level`);
      continue;
    }

    // --- Default emission ---
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
      case "gate_mul":
        lines.push(`        x = gate * x`);
        break;
      case "scale_dot": {
        const nh = resolve(p.cfg.numHeads || "heads", globalCfg);
        const hd = resolve(p.cfg.headDim || "headDim", globalCfg);
        lines.push(`        # Scaled dot-product attention (${nh} heads, dim ${hd})`);
        lines.push(`        x = F.scaled_dot_product_attention(q, k, v)`);
        break;
      }
      case "custom_note":
        lines.push(`        # ${p.cfg.text || "annotation"}`);
        break;
      default:
        lines.push(`        # ${p.type}: ${p.label}`);
    }
  }

  return lines;
}

// ─── Main Generator ───────────────────────────────────────────────────

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

  lines.push(`def apply_rope(x, seq_len):`);
  lines.push(`    """Simplified RoPE — replace with full implementation for production."""`);
  lines.push(`    d = x.shape[-1]`);
  lines.push(`    pos = torch.arange(seq_len, device=x.device).unsqueeze(1)`);
  lines.push(`    dim_idx = torch.arange(0, d, 2, device=x.device).float()`);
  lines.push(`    freqs = pos / (10000.0 ** (dim_idx / d))`);
  lines.push(`    cos_f, sin_f = freqs.cos(), freqs.sin()`);
  lines.push(`    x1, x2 = x[..., ::2], x[..., 1::2]`);
  lines.push(`    return torch.stack([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1).flatten(-2)`);
  lines.push(``);
  lines.push(``);

  // Collect ALL unique block signatures across all layers
  const blockClasses = new Map(); // sig -> { className, block, isParallel }

  // Determine per-layer topology info
  const layerTopologies = layers.map(l => {
    const topo = l.topology || "sequential";
    const pc = l.parallelCount || (topo !== "sequential" ? l.blocks.length : 0);
    return { topo, parallelCount: pc };
  });

  // Scan all layers for unique block classes
  layers.forEach((layer, li) => {
    const { topo, parallelCount } = layerTopologies[li];
    layer.blocks.forEach((block, bIdx) => {
      const isParallel = topo === "parallel" || (topo === "parallel_then_sequential" && bIdx < parallelCount);
      const sig = block.primitives.map(p => `${p.type}:${JSON.stringify(p.cfg)}`).join("|") + `|parallel:${isParallel}`;

      if (!blockClasses.has(sig)) {
        // Ensure unique class names
        let className = sanitize(block.name || `Block${bIdx}`);
        const existing = new Set([...blockClasses.values()].map(v => v.className));
        let suffix = 2;
        const base = className;
        while (existing.has(className)) { className = `${base}_${suffix++}`; }

        blockClasses.set(sig, { className, block, isParallel });
      }
    });
  });

  // Emit block classes
  for (const [, { className, block, isParallel }] of blockClasses) {
    const analysis = analyzeBlock(block.primitives, globalCfg);

    lines.push(`class ${className}(nn.Module):`);
    lines.push(`    def __init__(self, config: ModelConfig):`);
    lines.push(`        super().__init__()`);

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
    if (isParallel) {
      lines.push(`    def forward(self, x):`);
    } else {
      lines.push(`    def forward(self, x, residual=None):`);
    }

    const fwdLines = generateBlockForward(block.primitives, globalCfg, analysis, isParallel);
    fwdLines.forEach(l => lines.push(l));
    lines.push(`        return x`);
    lines.push(``);
    lines.push(``);
  }

  // Build per-layer block class name lists
  const layerBlockClassNames = layers.map((layer, li) => {
    const { topo, parallelCount } = layerTopologies[li];
    return layer.blocks.map((block, bIdx) => {
      const isParallel = topo === "parallel" || (topo === "parallel_then_sequential" && bIdx < parallelCount);
      const sig = block.primitives.map(p => `${p.type}:${JSON.stringify(p.cfg)}`).join("|") + `|parallel:${isParallel}`;
      return blockClasses.get(sig).className;
    });
  });

  // Check if all layers have the same block layout (for ModuleList sharing)
  const firstLayerNames = layerBlockClassNames[0] || [];
  const allLayersSame = layerBlockClassNames.every(names =>
    names.length === firstLayerNames.length && names.every((n, i) => n === firstLayerNames[i]));

  // Model class
  lines.push(`class TransformerModel(nn.Module):`);
  lines.push(`    def __init__(self, config: ModelConfig):`);
  lines.push(`        super().__init__()`);
  lines.push(`        self.config = config`);
  lines.push(`        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)`);

  if (firstLayerNames.length > 0) {
    if (allLayersSame) {
      lines.push(`        self.layers = nn.ModuleList([`);
      lines.push(`            nn.ModuleList([`);
      firstLayerNames.forEach(cn => lines.push(`                ${cn}(config),`));
      lines.push(`            ]) for _ in range(config.num_layers)`);
      lines.push(`        ])`);
    } else {
      lines.push(`        self.layers = nn.ModuleList([`);
      layerBlockClassNames.forEach((names, li) => {
        lines.push(`            nn.ModuleList([${names.map(n => `${n}(config)`).join(", ")}]),  # Layer ${li + 1}`);
      });
      lines.push(`        ])`);
    }
  }
  lines.push(`        self.norm = RMSNorm(config.hidden_dim)`);
  lines.push(`        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)`);
  lines.push(``);
  lines.push(`    def forward(self, input_ids):`);
  lines.push(`        x = self.embed(input_ids)`);

  if (firstLayerNames.length > 0) {
    // Determine the representative topology for code emission
    // If all layers share the same topology, emit a clean loop
    const allTopos = layerTopologies.map(t => t.topo);
    const uniformTopo = allTopos.every(t => t === allTopos[0]) ? allTopos[0] : null;

    if (uniformTopo === "sequential" || (!uniformTopo && !allTopos.some(t => t !== "sequential"))) {
      lines.push(`        for layer_blocks in self.layers:`);
      lines.push(`            residual = x`);
      lines.push(`            for block in layer_blocks:`);
      lines.push(`                x = block(x, residual=residual)`);
    } else if (uniformTopo === "parallel") {
      lines.push(`        for layer_blocks in self.layers:`);
      lines.push(`            residual = x`);
      lines.push(`            branch_outputs = [block(x) for block in layer_blocks]`);
      lines.push(`            x = residual + sum(branch_outputs)`);
    } else if (uniformTopo === "parallel_then_sequential") {
      const pc = layerTopologies[0].parallelCount;
      lines.push(`        for layer_blocks in self.layers:`);
      lines.push(`            residual = x`);
      lines.push(`            # Parallel branches (${pc} blocks)`);
      lines.push(`            branch_outputs = [layer_blocks[i](x) for i in range(${pc})]`);
      lines.push(`            x = residual + sum(branch_outputs)`);
      if (firstLayerNames.length > pc) {
        lines.push(`            # Sequential blocks`);
        lines.push(`            for block in layer_blocks[${pc}:]:`);
        lines.push(`                residual = x`);
        lines.push(`                x = block(x, residual=residual)`);
      }
    } else {
      // Mixed topologies — emit per-layer logic
      lines.push(`        for li, layer_blocks in enumerate(self.layers):`);
      // Group layers by topology
      const topoGroups = [];
      let currentGroup = null;
      allTopos.forEach((t, i) => {
        const pc = layerTopologies[i].parallelCount;
        const key = `${t}:${pc}`;
        if (!currentGroup || currentGroup.key !== key) {
          currentGroup = { key, topo: t, pc, start: i, end: i };
          topoGroups.push(currentGroup);
        } else {
          currentGroup.end = i;
        }
      });

      topoGroups.forEach(g => {
        const cond = g.start === g.end
          ? `li == ${g.start}`
          : `${g.start} <= li <= ${g.end}`;
        if (g.topo === "sequential") {
          lines.push(`            if ${cond}:`);
          lines.push(`                residual = x`);
          lines.push(`                for block in layer_blocks:`);
          lines.push(`                    x = block(x, residual=residual)`);
        } else if (g.topo === "parallel") {
          lines.push(`            if ${cond}:`);
          lines.push(`                residual = x`);
          lines.push(`                branch_outputs = [block(x) for block in layer_blocks]`);
          lines.push(`                x = residual + sum(branch_outputs)`);
        } else {
          lines.push(`            if ${cond}:`);
          lines.push(`                residual = x`);
          lines.push(`                branch_outputs = [layer_blocks[i](x) for i in range(${g.pc})]`);
          lines.push(`                x = residual + sum(branch_outputs)`);
          lines.push(`                for block in layer_blocks[${g.pc}:]:`);
          lines.push(`                    residual = x`);
          lines.push(`                    x = block(x, residual=residual)`);
        }
      });
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
