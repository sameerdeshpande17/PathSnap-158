export default function EvalPanel({ evalResult }) {
  if (!evalResult) {
    return (
      <div className="panel">
        <h2>Evaluation</h2>
        <button className="btn btn-secondary" style={{ width: "auto" }}>
          Run Evaluation
        </button>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Evaluation Results</h2>
      <p style={{ fontSize: 0.8, color: '#666', marginBottom: 12 }}>
        Weak-label agreement vs supplied reference route (not verified ground truth)
      </p>
      <div className="card">
        <div className="row">
          <span className="label">Classifier Top-1 agreement</span>
          <span className="value">
            {(evalResult.classifier_top1_agreement_rate * 100).toFixed(2)}%
          </span>
        </div>
        <div className="row">
          <span className="label">HMM/Viterbi agreement</span>
          <span className="value">
            {(evalResult.hmm_agreement_rate * 100).toFixed(2)}%
          </span>
        </div>
        <div className="row">
          <span className="label">Route precision</span>
          <span className="value">
            {(evalResult.route_precision * 100).toFixed(2)}%
          </span>
        </div>
        <div className="row">
          <span className="label">Route recall</span>
          <span className="value">
            {(evalResult.route_recall * 100).toFixed(2)}%
          </span>
        </div>
        <div className="row">
          <span className="label">Route F1</span>
          <span className="value">
            {(evalResult.route_f1 * 100).toFixed(2)}%
          </span>
        </div>
        <div className="row">
          <span className="label">Route connectivity</span>
          <span className="value">
            {(evalResult.route_connectivity * 100).toFixed(2)}%
          </span>
        </div>
        <div className="row">
          <span className="label">Edit distance</span>
          <span className="value">{evalResult.route_edit_distance}</span>
        </div>
      </div>
    </div>
  );
}
