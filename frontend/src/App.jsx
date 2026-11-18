import { useState, useEffect, useCallback } from "react";
import { checkHealth, listTrips, predictTrip, evaluateTrip } from "./api";
import ConnectivityStatus from "./components/ConnectivityStatus";
import TripSelector from "./components/TripSelector";
import MapView from "./components/MapView";
import ResultsPanel from "./components/ResultsPanel";
import EvalPanel from "./components/EvalPanel";
import MetricsPage from "./components/MetricsPage";

export default function App() {
  const [page, setPage] = useState(() => window.location.hash === "#/metrics" ? "metrics" : "predict");
  useEffect(() => {
    const onHashChange = () => setPage(window.location.hash === "#/metrics" ? "metrics" : "predict");
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  const [status, setStatus] = useState("unknown");
  const [statusError, setStatusError] = useState(null);
  const [trips, setTrips] = useState([]);
  const [selectedTrip, setSelectedTrip] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState(null);

  // Check backend health on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await checkHealth();
        if (!cancelled) {
          setStatus(res.status === "ready" ? "online" : "degraded");
          setStatusError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setStatus("offline");
          setStatusError(err.message);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load trips on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await listTrips("student_test");
        if (!cancelled) {
          setTrips(data);
          if (data.length > 0) {
            setSelectedTrip(data[0].trip_id);
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handlePredict = async () => {
    if (!selectedTrip) return;
    setPredicting(true);
    setError(null);
    setPrediction(null);
    setEvaluation(null);
    try {
      const res = await predictTrip(selectedTrip, "student_test");
      setPrediction(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setPredicting(false);
    }
  };

  const handleEvaluate = async () => {
    if (!selectedTrip) return;
    setEvaluating(true);
    setError(null);
    setPrediction(null);
    try {
      const res = await evaluateTrip(selectedTrip, "student_test");
      setEvaluation(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setEvaluating(false);
    }
  };

  const handleTripSelect = (tripId) => {
    setSelectedTrip(tripId);
    setPrediction(null);
    setEvaluation(null);
    setError(null);
  };

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>PathSnap</h1>
        </div>
        <ConnectivityStatus status={status} error={statusError} />
      </div>
      <nav className="page-nav" aria-label="Main navigation">
        <a href="#/predict" aria-current={page === "predict" ? "page" : undefined}>Prediction & Evaluation</a>
        <a href="#/metrics" aria-current={page === "metrics" ? "page" : undefined}>Model Metrics</a>
      </nav>

      {page === "metrics" ? <MetricsPage /> : <div className="main">
        <div className="sidebar">
          <div className="status-banner">
            {status === "online" ? (
              <span style={{ color: "#2e7d32", fontWeight: 600 }}>
                ● Backend connected
              </span>
            ) : status === "offline" ? (
              <span style={{ color: "#c62828", fontWeight: 600 }}>
                ● Backend offline
              </span>
            ) : (
              <span style={{ color: "#f57f17", fontWeight: 600 }}>
                ● Backend starting...
              </span>
            )}
          </div>

          <TripSelector onSelect={handleTripSelect} />

          <div style={{ marginTop: 12 }}>
            <button
              className="btn btn-primary"
              disabled={predicting || !selectedTrip || status !== "online"}
              onClick={handlePredict}
            >
              {predicting ? "Predicting..." : "Run Prediction"}
            </button>
            <button
              className="btn btn-secondary"
              disabled={evaluating || !selectedTrip || status !== "online"}
              onClick={handleEvaluate}
            >
              {evaluating ? "Evaluating..." : "Evaluate vs Reference"}
            </button>
          </div>

          {error && <div className="error-text">Error: {error}</div>}
        </div>

        <div className="content">
          <div className="map-container">
            <MapView
              gpsPoints={[]}
              predictedArcs={prediction?.arc_id_sequence ?? []}
              referenceArcs={evaluation ? [] : []}
              showReference={false}
            />
          </div>
          <div className="panel">
            <ResultsPanel result={prediction} />
            <EvalPanel evalResult={evaluation} />
          </div>
        </div>
      </div>}
    </div>
  );
}
