export default function ResultsPanel({ result }) {
  if (!result) {
    return (
      <div className="panel">
        <h2>Results</h2>
        <div style={{ fontSize: "0.875rem", color: "#666" }}>
          Run a prediction to see results here.
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Prediction Results</h2>
      <div className="card">
        <div className="row">
          <span className="label">Model</span>
          <span className="value">{result.model}</span>
        </div>
        <div className="row">
          <span className="label">Device</span>
          <span className="value">{result.device}</span>
        </div>
        <div className="row">
          <span className="label">Observations</span>
          <span className="value">{result.n_observations}</span>
        </div>
        <div className="row">
          <span className="label">Candidates</span>
          <span className="value">{result.n_candidates}</span>
        </div>
        <div className="row">
          <span className="label">Route segments</span>
          <span className="value">{result.route_n_segments}</span>
        </div>
        <div className="row">
          <span className="label">Route connectors</span>
          <span className="value">{result.route_n_connectors}</span>
        </div>
        <div className="row">
          <span className="label">Disconnected gaps</span>
          <span className="value">{result.route_disconnected_gaps}</span>
        </div>
        <div className="row">
          <span className="label">Route length (m)</span>
          <span className="value">{result.route_total_length_m?.toFixed(1)}</span>
        </div>
      </div>
      <div className="card">
        <h3>Timing</h3>
        <div className="row">
          <span className="label">Candidate generation</span>
          <span className="value">
            {result.timing_s?.candidate_gen?.toFixed(3)} s
          </span>
        </div>
        <div className="row">
          <span className="label">Classifier scoring</span>
          <span className="value">
            {result.timing_s?.classifier_score?.toFixed(3)} s
          </span>
        </div>
        <div className="row">
          <span className="label">Viterbi decoding</span>
          <span className="value">{result.timing_s?.viterbi?.toFixed(3)} s</span>
        </div>
        <div className="row">
          <span className="label">Route reconstruction</span>
          <span className="value">
            {result.timing_s?.route_reconstruct?.toFixed(3)} s
          </span>
        </div>
        <div className="row" style={{ fontWeight: 700, borderTop: "1px solid #ddd", paddingTop: 8 }}>
          <span className="label">Total</span>
          <span className="value">{result.timing_s?.total?.toFixed(3)} s</span>
        </div>
      </div>
      <div className="card">
        <h3>Predicted Arc IDs</h3>
        <div className="row">
          <span className="label">Sequence</span>
          <span className="value" style={{ fontSize: "0.75rem", wordBreak: "break-all" }}>
            {result.arc_id_sequence?.slice(0, 20).join(", ")}
            {result.arc_id_sequence.length > 20 ? ` … (+${result.arc_id_sequence.length - 20} more)` : ""}
          </span>
        </div>
      </div>
    </div>
  );
}
