import { useEffect, useState } from "react";
import { listTrips } from "../api";

export default function TripSelector({ onSelect }) {
  const [trips, setTrips] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await listTrips("student_test");
        if (!cancelled) {
          setTrips(data);
          if (!selected && data.length > 0) {
            setSelected(data[0].trip_id);
            if (onSelect) onSelect(data[0].trip_id);
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  return (
    <div className="card">
      <h3>Select Trip</h3>
      {loading ? (
        <div style={{ fontSize: "0.875rem", color: "#666" }}>Loading trips...</div>
      ) : error ? (
        <div className="error-text">Error loading trips: {error}</div>
      ) : (
        trips.map((t) => (
          <div
            key={t.trip_id}
            className="trip-item"
            style={{
              backgroundColor: selected === t.trip_id ? "#e3f2fd" : "#fff",
              borderColor: selected === t.trip_id ? "#2196f3" : "#e0e0e0",
            }}
            onClick={() => {
              setSelected(t.trip_id);
              if (onSelect) onSelect(t.trip_id);
            }}
          >
            <div className="trip-id">{t.trip_id}</div>
            <div className="trip-meta">
              {t.n_observations} obs • {t.n_nodes} nodes • {t.n_arcs} arcs
            </div>
          </div>
        ))
      )}
    </div>
  );
}
