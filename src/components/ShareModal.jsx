import { useRef, useState } from "react";
import { fmt } from "../utils/format";
import { buildExportPayload, parseImportData } from "../utils/exportImport";
import { S, font } from "./styles";

export default function ShareModal({
  archName, setArchName, archAuthor, setArchAuthor, archDesc, setArchDesc,
  stats, globalCfg, layers, onImport, onClose, showToast,
}) {
  const [importError, setImportError] = useState(null);
  const fileInputRef = useRef(null);

  const exportToFile = () => {
    const data = buildExportPayload({ archName, archAuthor, archDesc, stats, globalCfg, layers });
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safeName = (archName || "transformer").replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
    a.download = `${safeName}-${fmt(stats.totalP)}-${layers.length}L.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Exported to file");
  };

  const exportToClipboard = async () => {
    try {
      const data = buildExportPayload({ archName, archAuthor, archDesc, stats, globalCfg, layers });
      await navigator.clipboard.writeText(JSON.stringify(data));
      showToast("Copied to clipboard — share it!");
    } catch {
      const data = buildExportPayload({ archName, archAuthor, archDesc, stats, globalCfg, layers });
      const ta = document.createElement("textarea");
      ta.value = JSON.stringify(data);
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      showToast("Copied to clipboard — share it!");
    }
  };

  const doImport = (data) => {
    try {
      const result = parseImportData(data);
      onImport(result);
      setImportError(null);
      showToast(result.toastMsg);
      onClose();
    } catch (err) {
      setImportError(err.message);
      showToast(err.message, "error");
    }
  };

  const importFromFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        doImport(JSON.parse(ev.target.result));
      } catch {
        showToast("Invalid JSON file", "error");
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const copyLink = async () => {
    try {
      const data = buildExportPayload({ archName, archAuthor, archDesc, stats, globalCfg, layers });
      const json = JSON.stringify(data);
      const encoded = btoa(encodeURIComponent(json));
      const url = `${window.location.origin}${window.location.pathname}#config=${encoded}`;
      await navigator.clipboard.writeText(url);
      showToast("Shareable link copied!");
    } catch {
      showToast("Failed to copy link", "error");
    }
  };

  const importFromClipboard = async () => {
    try {
      const text = await navigator.clipboard.readText();
      doImport(JSON.parse(text));
    } catch {
      showToast("No valid config found on clipboard", "error");
    }
  };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(0,0,0,0.7)", backdropFilter: "blur(4px)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 16,
    }} onClick={onClose}>
      <div style={{
        ...S.panel, width: "100%", maxWidth: 420, maxHeight: "90vh", overflowY: "auto",
        padding: 0, background: "#0a0f1c", border: "1px solid #22c55e22",
        boxShadow: "0 0 40px rgba(34,197,94,0.08)",
      }} onClick={e => e.stopPropagation()}>
        {/* Modal header */}
        <div style={{
          padding: "12px 14px", borderBottom: "1px solid #172035",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <div style={{ fontSize: 7, color: "#22c55e", letterSpacing: "0.3em", textTransform: "uppercase", fontWeight: 600 }}>Share & Import</div>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#e2eaf2", marginTop: 2 }}>Architecture Config</div>
          </div>
          <button onClick={onClose} style={{
            background: "none", border: "none", color: "#3e5775", cursor: "pointer", fontSize: 16, fontWeight: 700, padding: "0 4px",
          }}>×</button>
        </div>

        {/* Metadata fields */}
        <div style={{ padding: "10px 14px", borderBottom: "1px solid #172035" }}>
          <div style={{ ...S.label, marginBottom: 6, color: "#06b6d4" }}>Architecture Metadata</div>
          <div style={{ marginBottom: 6 }}>
            <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Name</label>
            <input value={archName} onChange={e => setArchName(e.target.value)}
              placeholder="e.g. WobbleNet v3, Cerberus 56M Scout..."
              style={{
                width: "100%", background: "#060a14", border: "1px solid #172035", color: "#c8d6e5",
                fontFamily: font, fontSize: 10, padding: "5px 8px", borderRadius: 3,
              }} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 6 }}>
            <div>
              <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Author</label>
              <input value={archAuthor} onChange={e => setArchAuthor(e.target.value)}
                placeholder="Your name"
                style={{
                  width: "100%", background: "#060a14", border: "1px solid #172035", color: "#c8d6e5",
                  fontFamily: font, fontSize: 10, padding: "5px 8px", borderRadius: 3,
                }} />
            </div>
            <div>
              <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Params</label>
              <div style={{
                background: "#060a14", border: "1px solid #172035", borderRadius: 3,
                padding: "5px 8px", fontSize: 10, color: "#22c55e", fontWeight: 600,
              }}>{fmt(stats.totalP)} ({stats.numLayers}L)</div>
            </div>
          </div>
          <div>
            <label style={{ ...S.label, fontSize: 8, display: "block", marginBottom: 2 }}>Notes</label>
            <textarea value={archDesc} onChange={e => setArchDesc(e.target.value)} rows={2}
              placeholder="Design notes, rationale, what you changed..."
              style={{
                width: "100%", background: "#060a14", border: "1px solid #172035", color: "#c8d6e5",
                fontFamily: font, fontSize: 10, padding: "5px 8px", borderRadius: 3, resize: "vertical",
              }} />
          </div>
        </div>

        {/* Export section */}
        <div style={{ padding: "10px 14px", borderBottom: "1px solid #172035" }}>
          <div style={{ ...S.label, marginBottom: 8, color: "#22c55e" }}>Export</div>
          <div style={{ display: "flex", gap: 6 }}>
            <button onClick={exportToFile} style={{
              flex: 1, background: "linear-gradient(135deg, #22c55e, #16a34a)", color: "#000",
              border: "none", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 700, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ↓ Save File
            </button>
            <button onClick={exportToClipboard} style={{
              flex: 1, background: "linear-gradient(135deg, #06b6d4, #0891b2)", color: "#000",
              border: "none", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 700, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ⧉ Copy to Clipboard
            </button>
          </div>
          <button onClick={copyLink} style={{
            width: "100%", marginTop: 6, background: "#172035", color: "#c8d6e5",
            border: "1px solid #22354a", borderRadius: 4, padding: "8px 0", fontSize: 9,
            fontFamily: font, fontWeight: 600, cursor: "pointer",
            letterSpacing: "0.08em", textTransform: "uppercase",
          }}>
            ⌗ Copy Shareable Link
          </button>
          <div style={{ fontSize: 8, color: "#2a3f55", marginTop: 4, textAlign: "center" }}>
            Link encodes the full config — recipient clicks to load
          </div>
        </div>

        {/* Import section */}
        <div style={{ padding: "10px 14px" }}>
          <div style={{ ...S.label, marginBottom: 8, color: "#f97316" }}>Import</div>
          <div style={{ display: "flex", gap: 6 }}>
            <button onClick={() => fileInputRef.current?.click()} style={{
              flex: 1, background: "#172035", color: "#c8d6e5",
              border: "1px solid #22354a", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 600, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ↑ Load File
            </button>
            <button onClick={importFromClipboard} style={{
              flex: 1, background: "#172035", color: "#c8d6e5",
              border: "1px solid #22354a", borderRadius: 4, padding: "8px 0", fontSize: 9,
              fontFamily: font, fontWeight: 600, cursor: "pointer",
              letterSpacing: "0.08em", textTransform: "uppercase",
            }}>
              ⧉ Paste from Clipboard
            </button>
          </div>
          <input ref={fileInputRef} type="file" accept=".json,application/json" onChange={importFromFile}
            style={{ display: "none" }} />
          {importError && (
            <div style={{ fontSize: 9, color: "#ef4444", marginTop: 6, padding: "4px 8px", background: "#1a0a0a", borderRadius: 3, border: "1px solid #2a1515" }}>
              {importError}
            </div>
          )}
          <div style={{ fontSize: 8, color: "#2a3f55", marginTop: 4, textAlign: "center" }}>
            Import replaces current architecture. Export first to save your work.
          </div>
        </div>
      </div>
    </div>
  );
}
