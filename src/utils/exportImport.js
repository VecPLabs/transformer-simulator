import { PRIM } from "../data/primitives";
import { uid } from "./uid";
import { fmt } from "./format";

export function buildExportPayload({ archName, archAuthor, archDesc, stats, globalCfg, layers }) {
  return {
    _format: "vecplabs-transformer-sim",
    _version: 2,
    meta: {
      name: archName || "Untitled",
      author: archAuthor || "Anonymous",
      description: archDesc || "",
      exportedAt: new Date().toISOString(),
      totalParams: stats.totalP,
      numLayers: layers.length,
    },
    globalConfig: globalCfg,
    layers: layers.map((l, i) => ({
      layer: i + 1,
      blocks: l.blocks.map(b => ({
        name: b.name,
        color: b.color,
        primitives: b.primitives.map(p => ({
          type: p.type,
          label: p.label,
          config: p.cfg,
        })),
      })),
    })),
    stats: {
      totalParams: stats.totalP,
      embeddingParams: stats.embP,
      outputParams: stats.outP,
      layerParams: stats.layerP,
      totalFlops: stats.totalFlops || 0,
      primitiveCounts: Object.entries(stats.primCounts).map(([k, v]) => ({
        type: k, name: PRIM[k]?.name, count: v.count, totalParams: v.params,
      })),
    },
  };
}

export function parseImportData(data) {
  // Validate format
  if (data._format !== "vecplabs-transformer-sim") {
    if (!(data.globalConfig && data.layers)) {
      throw new Error("Unrecognized file format. Expected a VecP Labs Transformer Simulator export.");
    }
  }

  // Parse global config
  const gc = data.globalConfig || {};
  const globalCfg = {
    hiddenDim: gc.hiddenDim || 2048,
    ffnDim: gc.ffnDim || 5504,
    heads: gc.heads || 16,
    kvHeads: gc.kvHeads || 4,
    vocabSize: gc.vocabSize || 32000,
    seqLen: gc.seqLen || 4096,
    moeExperts: gc.moeExperts || 8,
    moeTopK: gc.moeTopK || 2,
  };

  // Parse layers
  let layers = [];
  if (data.layers && Array.isArray(data.layers)) {
    layers = data.layers.map(layerData => {
      const blocks = (layerData.blocks || []).map(blockData => {
        const primitives = (blockData.primitives || []).map(primData => {
          if (!PRIM[primData.type]) {
            console.warn(`Unknown primitive type: ${primData.type}, skipping`);
            return null;
          }
          return {
            id: uid(),
            type: primData.type,
            label: primData.label || PRIM[primData.type].name,
            cfg: primData.config || { ...PRIM[primData.type].defaultCfg },
          };
        }).filter(Boolean);

        return {
          id: uid(),
          name: blockData.name || "Imported Block",
          color: blockData.color || "#334155",
          collapsed: false,
          primitives,
        };
      });

      return { id: uid(), collapsed: false, blocks };
    });
  }

  // Parse metadata
  const meta = {
    name: data.meta?.name || "Imported Architecture",
    author: data.meta?.author || "",
    desc: data.meta?.description || "",
  };

  const paramStr = data.meta?.totalParams ? ` (${fmt(data.meta.totalParams)} params)` : "";
  const toastMsg = `Imported "${meta.name}"${paramStr}`;

  return { globalCfg, layers, meta, toastMsg };
}
