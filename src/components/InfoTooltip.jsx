import { useState, useRef } from "react";
import { PRIM } from "../data/primitives";
import { font } from "./styles";

export default function InfoTooltip({ text, color }) {
  const [show, setShow] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const btnRef = useRef(null);

  const handleEnter = () => {
    if (btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setPos({
        top: rect.bottom + 6,
        left: Math.max(8, Math.min(rect.left - 100, window.innerWidth - 260)),
      });
    }
    setShow(true);
  };

  return (
    <>
      <button
        ref={btnRef}
        onMouseEnter={handleEnter}
        onMouseLeave={() => setShow(false)}
        onClick={e => { e.stopPropagation(); setShow(s => !s); }}
        style={{
          background: "none", border: "none", cursor: "pointer",
          fontSize: 8, padding: "0 2px", color: show ? color : "#3e5775",
          fontWeight: 700, lineHeight: 1, flexShrink: 0,
          transition: "color 0.12s",
        }}
      >?</button>
      {show && (
        <div style={{
          position: "fixed", top: pos.top, left: pos.left, zIndex: 999,
          width: 240, padding: "8px 10px",
          background: "#0c1220", border: `1px solid ${color}33`,
          borderRadius: 5, boxShadow: `0 4px 16px rgba(0,0,0,0.5), 0 0 8px ${color}10`,
          pointerEvents: "none",
        }}>
          <div style={{ fontSize: 9, fontWeight: 600, color, marginBottom: 3, fontFamily: font }}>{PRIM[text.type]?.icon} {PRIM[text.type]?.name}</div>
          <div style={{ fontSize: 9, color: "#8899aa", lineHeight: 1.5, fontFamily: font }}>{text.info}</div>
          <div style={{ fontSize: 8, color: "#2a3f55", marginTop: 4, fontStyle: "italic", fontFamily: font }}>{text.desc}</div>
        </div>
      )}
    </>
  );
}
