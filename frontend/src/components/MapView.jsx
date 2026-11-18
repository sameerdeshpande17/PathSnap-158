import { useEffect } from "react";
import { MapContainer, TileLayer, Polyline, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";

function TileLayerRef() {
  return (
    <TileLayer
      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
    />
  );
}

export default function MapView({
  gpsPoints,
  predictedArcs,
  referenceArcs,
  showReference,
}) {
  return (
    <MapContainer
      center={[49.844, 8.595]}
      zoom={13}
      style={{ height: "100%", width: "100%" }}
    >
      <TileLayerRef />
      {gpsPoints.length > 0 && (
        <Polyline
          positions={gpsPoints.map((p) => [p[1], p[0]])}
          color="#2980b9"
          weight={3}
          opacity={0.8}
        >
          <Popup>GPS observations ({gpsPoints.length} points)</Popup>
        </Polyline>
      )}
      {/* predictedArcs is an array of arc IDs (integers), not coordinates.
          We cannot draw them without node coordinates. Show info instead. */}
      {predictedArcs.length > 0 && (
        <div
          style={{
            position: "absolute",
            bottom: 20,
            left: 20,
            zIndex: 1000,
            background: "rgba(255, 255, 255, 0.9)",
            padding: "10px 14px",
            borderRadius: 6,
            border: "1px solid #e74c3c",
            color: "#c62828",
            fontSize: "0.85rem",
            fontWeight: 600,
            maxWidth: "300px",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6, color: "#c62828" }}>
            Predicted Route ({predictedArcs.length} arcs)
          </div>
          <div style={{ fontSize: "0.75rem", fontFamily: "monospace" }}>
            {predictedArcs.slice(0, 10).join(", ")}
            {predictedArcs.length > 10 ? ` … (+${predictedArcs.length - 10} more)` : ""}
          </div>
        </div>
      )}
    </MapContainer>
  );
}
