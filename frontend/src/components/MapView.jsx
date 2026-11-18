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
      zoom={14}
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
      {predictedArcs.length > 0 && (
        <Polyline
          positions={predictedArcs.map((p) => [p[1], p[0]])}
          color="#e74c3c"
          weight={4}
          opacity={0.9}
        >
          <Popup>Predicted route ({predictedArcs.length} arcs)</Popup>
        </Polyline>
      )}
      {showReference && referenceArcs.length > 0 && (
        <Polyline
          positions={referenceArcs.map((p) => [p[1], p[0]])}
          color="#2ecc71"
          weight={4}
          opacity={0.5}
          dashArray="10, 5"
        >
          <Popup>Reference route ({referenceArcs.length} arcs)</Popup>
        </Polyline>
      )}
    </MapContainer>
  );
}
