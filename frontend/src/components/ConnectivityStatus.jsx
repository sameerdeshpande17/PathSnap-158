import { useEffect, useState } from "react";
import { checkHealth } from "../api";

export default function ConnectivityStatus() {
  const [status, setStatus] = useState("unknown");
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await checkHealth();
        if (!cancelled) {
          setStatus(res.status === "ready" ? "online" : "degraded");
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setStatus("offline");
          setError(err.message);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const color =
    status === "online" ? "#2ecc71" : status === "offline" ? "#e74c3c" : "#f39c12";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 12px",
        borderRadius: 6,
        background:
          status === "online"
            ? "#e8f5e9"
            : status === "offline"
            ? "#ffebee"
            : "#fffde7",
        border: `1px solid ${color}`,
        color: status === "online" ? "#2e7d32" : status === "offline" ? "#c62828" : "#f57f17",
        fontSize: "0.85rem",
        fontWeight: 600,
      }}
    >
      <div
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: color,
        }}
      />
      {status === "online"
        ? "Backend connected"
        : status === "offline"
        ? `Backend offline${error ? ": " + error : ""}`
        : "Backend starting..."}
    </div>
  );
}
