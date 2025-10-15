import React, { useState } from "react";

export default function ScoreForm() {
  const [raw, setRaw] = useState("GET /product?id=100' OR '1'='1 HTTP/1.1");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [threshold, setThreshold] = useState(null);

  // optional: fetch threshold from a static file or backend endpoint
  // For now we'll leave it empty; you can set threshold manually for quick checks.

  async function handleSubmit(e) {
    e?.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // call backend /score (dev proxy will forward to http://localhost:8000)
      const resp = await fetch("/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw })
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }

      const json = await resp.json();
      setResult(json);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 720 }}>
      <form onSubmit={handleSubmit}>
        <label style={{ display: "block", marginBottom: 8 }}>
          Raw request
        </label>
        <textarea
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          rows={4}
          style={{ width: "100%", padding: 8, fontFamily: "monospace" }}
        />
        <div style={{ marginTop: 12 }}>
          <button type="submit" disabled={loading}>
            {loading ? "Scoring..." : "Score request"}
          </button>
          <button
            type="button"
            onClick={() =>
              setRaw("GET /product?id=100' OR '1'='1 HTTP/1.1")
            }
            style={{ marginLeft: 8 }}
          >
            Use sample
          </button>
        </div>
      </form>

      <div style={{ marginTop: 16 }}>
        {error && (
          <div style={{ color: "crimson" }}>
            <strong>Error:</strong> {error}
          </div>
        )}
        {result && (
          <div style={{ marginTop: 12 }}>
            <h3>Result</h3>
            <pre style={{ whiteSpace: "pre-wrap" }}>
              {JSON.stringify(result, null, 2)}
            </pre>
            {typeof result.score === "number" && threshold !== null && (
              <div>
                <strong>Anomaly:</strong>{" "}
                {result.score > threshold ? "YES" : "NO"} (threshold: {threshold})
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
