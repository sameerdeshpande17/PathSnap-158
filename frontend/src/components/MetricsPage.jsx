import selection from "../data/model-selection.json";

const names = {
  hist_gradient_boosting: "Histogram Gradient Boosting",
  random_forest: "Random Forest",
  mlp: "Multilayer Perceptron",
};
const percent = (value) => `${(value * 100).toFixed(4)}%`;

export default function MetricsPage() {
  const best = selection.comparison[0];
  return (
    <main className="metrics-page">
      <h1>Model Metrics</h1>
      <p>Saved experiment results · {selection.basis}</p>
      <div className="metrics-grid">
        <section className="card">
          <h2>Selected model</h2>
          <strong>{names[selection.recommended_model]}</strong>
          <p>Optuna trial {selection.recommended_from_trial}</p>
        </section>
        <section className="card">
          <h2>CV Top-1 agreement</h2>
          <strong>{percent(best.best_top1)}</strong>
          <p>Agreement with weak reference labels, not independently verified GPS accuracy.</p>
        </section>
        <section className="card">
          <h2>Mean CV log loss</h2>
          <strong>{best.mean_logloss.toFixed(6)}</strong>
          <p>Lower is better.</p>
        </section>
      </div>
      <section className="card">
        <h2>Model comparison</h2>
        <div className="metrics-table-wrap">
          <table>
            <caption>Saved training cross-validation scores</caption>
            <thead><tr><th scope="col">Model</th><th scope="col">Mean Top-1</th><th scope="col">Top-1 standard deviation</th><th scope="col">Mean log loss</th></tr></thead>
            <tbody>{selection.comparison.map((row) => (
              <tr key={row.model}>
                <th scope="row">{names[row.model]}{row.model === selection.recommended_model ? " (selected)" : ""}</th>
                <td>{percent(row.best_top1)}</td>
                <td>{(row.std_top1 * 100).toFixed(4)} percentage points</td>
                <td>{row.mean_logloss.toFixed(6)}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
      <section className="card">
        <h2>Why this model?</h2>
        <p>Histogram Gradient Boosting has the highest saved mean Top-1 score, but is effectively tied with Random Forest
          (a difference of {((best.best_top1 - selection.comparison[1].best_top1) * 100).toFixed(8)} percentage points).
          The saved selection favors the simpler model. Random Forest has slightly lower log loss.
          These results do not establish a meaningful accuracy advantage.</p>
      </section>
      <section className="card">
        <h2>What these results cover</h2>
        <p>These scores describe the tuned model-selection experiments, not the currently loaded backend checkpoint.
          Deployment of trial 49 has not been verified.</p>
        <p>Verified held-out accuracy, F1, MRR, candidate recall, and route continuity are unavailable in this snapshot.</p>
        <p>Source: <code>{selection.source}</code>. This bundled snapshot is available even when the backend is offline.</p>
      </section>
    </main>
  );
}
