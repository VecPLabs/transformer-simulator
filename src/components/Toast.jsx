import { font } from "./styles";

export default function Toast({ toast }) {
  if (!toast) return null;
  const isErr = toast.type === "error";
  return (
    <div style={{
      position: "fixed", bottom: 20, left: "50%", transform: "translateX(-50%)",
      zIndex: 1001, padding: "8px 16px", borderRadius: 6,
      background: isErr ? "#1a0a0a" : "#0a1a0a",
      border: `1px solid ${isErr ? "#ef444433" : "#22c55e33"}`,
      color: isErr ? "#ef4444" : "#22c55e",
      fontFamily: font, fontSize: 10, fontWeight: 600,
      boxShadow: `0 4px 20px ${isErr ? "rgba(239,68,68,0.15)" : "rgba(34,197,94,0.15)"}`,
      animation: "toastIn 0.2s ease",
      maxWidth: "90vw", textAlign: "center",
    }}>
      {isErr ? "✕ " : "✓ "}{toast.msg}
    </div>
  );
}
