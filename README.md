# Transformer Simulator

**Interactive transformer architecture builder.** Design, visualize, and share custom model architectures at the primitive level — from attention projections to activation functions.

Built by [VecP Labs](https://vecplabs.com).

---

## What Is This?

A visual tool for exploring transformer architectures. Instead of treating attention blocks and FFN layers as black boxes, this simulator breaks everything down into **primitives** — individual linear projections, activations, normalization ops, and connections — that you can add, remove, reorder, and configure.

Think of it as a circuit board designer for neural network architectures.

### Who Is This For?

- **Researchers** exploring non-standard architectures (double-attention, hybrid MoE, custom gating)
- **Students** learning how transformer components fit together
- **Engineers** planning model designs before committing to training runs
- **Anyone** curious about what's actually inside these models

## Features

### Primitive-Level Editing
Every component is built from atomic operations:
- **Projections:** Linear, Linear + Bias (configurable input/output dimensions)
- **Activations:** SiLU/Swish, GELU, ReLU, Softmax
- **Operations:** Gate Multiply, Residual Add, Scaled Dot-Product, RoPE, Concat, Split/Branch, MoE Router
- **Normalization:** RMSNorm, LayerNorm
- **Other:** Dropout, Annotations

### Architecture Building
- **Hierarchy:** Layers → Blocks → Primitives
- **Drag-and-drop** reordering within blocks
- **Block templates** for common patterns (Pre-Norm + GQA + Residual, SwiGLU FFN, etc.)
- **Build from scratch** with empty blocks
- **Per-primitive configuration** — set dimensions via named references (Hidden Dim, FFN Dim, Head Dim, etc.)

### Live Statistics
- Total parameter count with per-layer breakdown
- Primitive-type distribution across all layers
- F16 and Q4 model size estimates
- Architecture fingerprint (FFN/hidden ratio, head dim, Q/KV ratio, Chinchilla token count)

### Import / Export / Share
- **Export to JSON file** or **copy to clipboard**
- **Import from file** or **paste from clipboard**
- **Architecture metadata** — name, author, notes travel with the config
- **Versioned format** for forward compatibility
- Round-trip workflow: export → share → import → modify → export back

### Presets
- Llama-style (Pre-Norm + GQA + SwiGLU)
- GPT-2 style (Pre-Norm + MHA + GELU FFN)
- Bare Minimum (no norms, stripped down)

Load a preset and start modifying, or build from zero.

## Quick Start

### Use Online

Visit the hosted version (no install required):

> **[https://vecplabs.github.io/transformer-simulator/](https://vecplabs.github.io/transformer-simulator/)**

### Run Locally

```bash
git clone https://github.com/vecplabs/transformer-simulator.git
cd transformer-simulator
npm install
npm run dev
```

Opens at `http://localhost:3000`.

### Build for Production

```bash
npm run build
npm run preview
```

Static output goes to `dist/` — deploy anywhere.

## Project Structure

```
transformer-simulator/
├── src/
│   ├── main.jsx                  # Entry point
│   ├── App.jsx                   # App wrapper
│   └── TransformerSimulator.jsx  # Main simulator component
├── public/
│   └── favicon.svg
├── .github/
│   └── workflows/
│       └── deploy.yml            # Auto-deploy to GitHub Pages
├── index.html
├── vite.config.js
├── package.json
└── LICENSE                       # MIT
```

## Config Format

Exported configs use a versioned JSON format:

```json
{
  "_format": "vecplabs-transformer-sim",
  "_version": 2,
  "meta": {
    "name": "My Custom Architecture",
    "author": "Your Name",
    "description": "Design notes...",
    "exportedAt": "2026-03-29T...",
    "totalParams": 1234567890,
    "numLayers": 16
  },
  "globalConfig": {
    "hiddenDim": 2048,
    "ffnDim": 5504,
    "heads": 16,
    "kvHeads": 4,
    "vocabSize": 32000,
    "seqLen": 4096,
    "moeExperts": 8,
    "moeTopK": 2
  },
  "layers": [
    {
      "layer": 1,
      "blocks": [
        {
          "name": "Pre-Norm + GQA + Residual",
          "color": "#14b8a6",
          "primitives": [
            { "type": "rmsnorm", "label": "Attn Norm", "config": { "dim": "hiddenDim" } },
            { "type": "linear", "label": "W_Q", "config": { "inDim": "hiddenDim", "outDim": "hiddenDim" } },
            ...
          ]
        }
      ]
    }
  ]
}
```

Primitive configs use **named dimension references** (`"hiddenDim"`, `"ffnDim"`, `"headDim"`, etc.) so configs stay valid when you change global dimensions.

## Deployment

### GitHub Pages (Automatic)

The included GitHub Actions workflow (`.github/workflows/deploy.yml`) auto-deploys on push to `main`.

**Setup:**
1. Go to your repo → Settings → Pages
2. Set Source to **GitHub Actions**
3. Push to `main` — it builds and deploys automatically

### Manual Deploy Anywhere

```bash
npm run build
# Upload dist/ to any static host (Netlify, Vercel, Cloudflare Pages, S3, etc.)
```

If deploying to a subpath, update `base` in `vite.config.js`:
```js
base: '/your-subpath/'
```

If deploying to a root domain, set:
```js
base: '/'
```

## Contributing

PRs welcome. Some areas that could use work:

- [ ] Diff view for comparing two architectures
- [ ] Parameter budget mode (set target, see what fits)
- [ ] FLOP estimation per-primitive
- [ ] PyTorch / JAX config export
- [ ] Undo/redo
- [ ] Mobile layout improvements

## License

MIT — see [LICENSE](LICENSE).

---

**VecP Labs** — Independent AI research.
