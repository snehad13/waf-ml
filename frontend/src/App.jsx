import React from "react";
import ScoreForm from "./components/ScoreForm";

export default function App() {
  return (
    <div style={{ fontFamily: "system-ui, Arial", padding: 24 }}>
      <h1>WAF Scorer — Demo</h1>
      <p>Enter a raw request or use the sample button to test scoring.</p>
      <ScoreForm />
    </div>
  );
}
