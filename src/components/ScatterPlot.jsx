import { useMemo } from "react";
import { fmt } from "../utils/format";
import { S, font } from "./styles";

// Known reference architectures: [name, params, flops_per_token (approx)]
const REFERENCE_MODELS = [
  { name: "SmolLM 135M", params: 135e6, flops: 270e6, color: "#6b8299" },
  { name: "GPT-2 Small", params: 124e6, flops: 248e6, color: "#6b8299" },
  { name: "GPT-2 Medium", params: 355e6, flops: 710e6, color: "#6b8299" },
  { name: "GPT-2 Large", params: 774e6, flops: 1.55e9, color: "#6b8299" },
  { name: "Phi-3 Mini", params: 3.8e9, flops: 7.6e9, color: "#a78bfa" },
  { name: "Llama 3.2 1B", params: 1.24e9, flops: 2.48e9, color: "#06b6d4" },
  { name: "Llama 3.2 3B", params: 3.21e9, flops: 6.42e9, color: "#06b6d4" },
  { name: "Llama 3.1 8B", params: 8.03e9, flops: 16.06e9, color: "#06b6d4" },
  { name: "Qwen2.5 0.5B", params: 494e6, flops: 988e6, color: "#f97316" },
  { name: "Qwen2.5 3B", params: 3.09e9, flops: 6.18e9, color: "#f97316" },
  { name: "Mistral 7B", params: 7.24e9, flops: 14.48e9, color: "#ec4899" },
];

function logScale(val, min, max, size) {
  if (val <= 0) return 0;
  const logMin = Math.log10(Math.max(min, 1));
  const logMax = Math.log10(max);
  const logVal = Math.log10(val);
  return ((logVal - logMin) / (logMax - logMin)) * size;
}

export default function ScatterPlot({ stats }) {
  const { totalP: currentParams, totalFlops: currentFlops } = stats;

  const allPoints = useMemo(() => {
    const points = REFERENCE_MODELS.map(m => ({ ...m, isCurrent: false }));
    points.push({ name: "Your Model", params: currentParams, flops: currentFlops, color: "#22c55e", isCurrent: true });
    return points;
  }, [currentParams, currentFlops]);

  const bounds = useMemo(() => {
    const allParams = allPoints.map(p => p.params).filter(p => p > 0);
    const allFlops = allPoints.map(p => p.flops).filter(f => f > 0);
    return {
      minP: Math.min(...allParams) * 0.5,
      maxP: Math.max(...allParams) * 2,
      minF: Math.min(...allFlops) * 0.5,
      maxF: Math.max(...allFlops) * 2,
    };
  }, [allPoints]);

  const W = 320, H = 200, PAD = 30, PADR = 10, PADT = 10, PADB = 24;
  const plotW = W - PAD - PADR;
  const plotH = H - PADT - PADB;

  const toX = (p) => PAD + logScale(p, bounds.minP, bounds.maxP, plotW);
  const toY = (f) => PADT + plotH - logScale(f, bounds.minF, bounds.maxF, plotH);

  // Axis ticks
  const paramTicks = useMemo(() => {
    const ticks = [];
    const logMin = Math.floor(Math.log10(bounds.minP));
    const logMax = Math.ceil(Math.log10(bounds.maxP));
    for (let e = logMin; e <= logMax; e++) {
      const val = Math.pow(10, e);
      if (val >= bounds.minP && val <= bounds.maxP) ticks.push(val);
    }
    return ticks;
  }, [bounds]);

  const flopTicks = useMemo(() => {
    const ticks = [];
    const logMin = Math.floor(Math.log10(bounds.minF));
    const logMax = Math.ceil(Math.log10(bounds.maxF));
    for (let e = logMin; e <= logMax; e++) {
      const val = Math.pow(10, e);
      if (val >= bounds.minF && val <= bounds.maxF) ticks.push(val);
    }
    return ticks;
  }, [bounds]);

  return (
    <div style={{ ...S.panel, padding: "8px 10px", marginBottom: 8 }}>
      <div style={{ ...S.label, marginBottom: 8, color: "#ec4899" }}>Architecture Landscape</div>
      <svg width={W} height={H} style={{ display: "block", margin: "0 auto" }}>
        {/* Grid */}
        {paramTicks.map(v => {
          const x = toX(v);
          return <line key={`gx${v}`} x1={x} y1={PADT} x2={x} y2={PADT + plotH} stroke="#172035" strokeWidth={0.5} />;
        })}
        {flopTicks.map(v => {
          const y = toY(v);
          return <line key={`gy${v}`} x1={PAD} y1={y} x2={PAD + plotW} y2={y} stroke="#172035" strokeWidth={0.5} />;
        })}

        {/* Axes */}
        <line x1={PAD} y1={PADT + plotH} x2={PAD + plotW} y2={PADT + plotH} stroke="#2a3f55" strokeWidth={1} />
        <line x1={PAD} y1={PADT} x2={PAD} y2={PADT + plotH} stroke="#2a3f55" strokeWidth={1} />

        {/* Axis labels */}
        <text x={PAD + plotW / 2} y={H - 2} textAnchor="middle" fill="#3e5775" fontSize={7} fontFamily={font}>Parameters</text>
        <text x={6} y={PADT + plotH / 2} textAnchor="middle" fill="#3e5775" fontSize={7} fontFamily={font} transform={`rotate(-90, 6, ${PADT + plotH / 2})`}>FLOPs/tok</text>

        {/* Tick labels */}
        {paramTicks.map(v => (
          <text key={`tx${v}`} x={toX(v)} y={PADT + plotH + 10} textAnchor="middle" fill="#2a3f55" fontSize={6} fontFamily={font}>{fmt(v)}</text>
        ))}
        {flopTicks.map(v => (
          <text key={`ty${v}`} x={PAD - 3} y={toY(v) + 3} textAnchor="end" fill="#2a3f55" fontSize={6} fontFamily={font}>{fmt(v)}</text>
        ))}

        {/* Reference points */}
        {allPoints.filter(p => !p.isCurrent && p.flops > 0).map((pt, i) => {
          const x = toX(pt.params);
          const y = toY(pt.flops);
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={3} fill={pt.color} opacity={0.6} />
              <text x={x + 5} y={y + 3} fill={pt.color} fontSize={5.5} fontFamily={font} opacity={0.7}>{pt.name}</text>
            </g>
          );
        })}

        {/* Current model point */}
        {currentFlops > 0 && (
          <g>
            <circle cx={toX(currentParams)} cy={toY(currentFlops)} r={5} fill="#22c55e" opacity={0.9} />
            <circle cx={toX(currentParams)} cy={toY(currentFlops)} r={8} fill="none" stroke="#22c55e" strokeWidth={1} opacity={0.4} />
            <text x={toX(currentParams) + 8} y={toY(currentFlops) + 3} fill="#22c55e" fontSize={7} fontFamily={font} fontWeight={700}>You</text>
          </g>
        )}
      </svg>
      <div style={{ fontSize: 7, color: "#2a3f55", textAlign: "center", marginTop: 4 }}>
        Log scale · Reference models are approximate
      </div>
    </div>
  );
}
