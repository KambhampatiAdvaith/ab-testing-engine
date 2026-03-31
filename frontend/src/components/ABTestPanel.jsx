import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { runFrequentist, runBayesian } from "../api";

const field = (label, name, defaultVal) => ({ label, name, defaultVal });

const FREQ_FIELDS = [
  field("Control Clicks", "control_clicks", 120),
  field("Control Total", "control_total", 1000),
  field("Variant Clicks", "variant_clicks", 145),
  field("Variant Total", "variant_total", 1000),
  field("Alpha (α)", "alpha", 0.05),
];

const BAYES_FIELDS = [
  field("Control Successes", "control_successes", 120),
  field("Control Failures", "control_failures", 880),
  field("Variant Successes", "variant_successes", 145),
  field("Variant Failures", "variant_failures", 855),
  field("Prior α", "prior_alpha", 1),
  field("Prior β", "prior_beta", 1),
  field("Simulations", "n_simulations", 10000),
];

function InputField({ label, name, value, onChange }) {
  return (
    <label className="block">
      <span className="text-xs text-surface-400">{label}</span>
      <input
        type="number"
        name={name}
        value={value}
        onChange={onChange}
        step="any"
        className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
      />
    </label>
  );
}

function ResultRow({ label, value }) {
  return (
    <tr className="border-b border-surface-700/50">
      <td className="py-2 pr-4 text-sm text-surface-400">{label}</td>
      <td className="py-2 text-sm text-surface-300 font-mono">{value}</td>
    </tr>
  );
}

export default function ABTestPanel() {
  const [freqForm, setFreqForm] = useState(
    Object.fromEntries(FREQ_FIELDS.map((f) => [f.name, f.defaultVal]))
  );
  const [bayesForm, setBayesForm] = useState(
    Object.fromEntries(BAYES_FIELDS.map((f) => [f.name, f.defaultVal]))
  );
  const [freqResult, setFreqResult] = useState(null);
  const [bayesResult, setBayesResult] = useState(null);
  const [loading, setLoading] = useState({ freq: false, bayes: false });
  const [error, setError] = useState({ freq: null, bayes: null });

  const handleFreq = async (e) => {
    e.preventDefault();
    setLoading((s) => ({ ...s, freq: true }));
    setError((s) => ({ ...s, freq: null }));
    try {
      const parsed = Object.fromEntries(
        Object.entries(freqForm).map(([k, v]) => [k, Number(v)])
      );
      const res = await runFrequentist(parsed);
      setFreqResult(res);
    } catch (err) {
      setError((s) => ({
        ...s,
        freq: err.response?.data?.detail || err.message,
      }));
    } finally {
      setLoading((s) => ({ ...s, freq: false }));
    }
  };

  const handleBayes = async (e) => {
    e.preventDefault();
    setLoading((s) => ({ ...s, bayes: true }));
    setError((s) => ({ ...s, bayes: null }));
    try {
      const parsed = Object.fromEntries(
        Object.entries(bayesForm).map(([k, v]) => [k, Number(v)])
      );
      const res = await runBayesian(parsed);
      setBayesResult(res);
    } catch (err) {
      setError((s) => ({
        ...s,
        bayes: err.response?.data?.detail || err.message,
      }));
    } finally {
      setLoading((s) => ({ ...s, bayes: false }));
    }
  };

  const onChangeFreq = (e) =>
    setFreqForm((s) => ({ ...s, [e.target.name]: e.target.value }));
  const onChangeBayes = (e) =>
    setBayesForm((s) => ({ ...s, [e.target.name]: e.target.value }));

  const probBarData = bayesResult
    ? [
        { name: "P(B > A)", value: bayesResult.prob_b_better ?? bayesResult.probability_b_better ?? 0 },
        { name: "P(A ≥ B)", value: 1 - (bayesResult.prob_b_better ?? bayesResult.probability_b_better ?? 0) },
      ]
    : [];

  return (
    <div className="space-y-8 max-w-4xl">
      <h2 className="text-xl font-semibold text-surface-300">A/B Testing</h2>

      {/* Frequentist */}
      <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
        <h3 className="text-sm font-medium text-surface-300 mb-4">
          Frequentist Z-Test
        </h3>
        <form onSubmit={handleFreq} className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {FREQ_FIELDS.map((f) => (
            <InputField
              key={f.name}
              label={f.label}
              name={f.name}
              value={freqForm[f.name]}
              onChange={onChangeFreq}
            />
          ))}
          <div className="col-span-full">
            <button
              type="submit"
              disabled={loading.freq}
              className="px-4 py-2 rounded-md bg-accent-500 text-white text-sm font-medium hover:bg-accent-400 disabled:opacity-50 transition-colors cursor-pointer"
            >
              {loading.freq ? "Running…" : "Run Test"}
            </button>
          </div>
        </form>

        {error.freq && (
          <p className="mt-3 text-sm text-danger-400">{error.freq}</p>
        )}

        {freqResult && (
          <table className="mt-4 w-full">
            <tbody>
              <ResultRow label="Z-Statistic" value={freqResult.z_stat?.toFixed(4)} />
              <ResultRow label="P-Value" value={freqResult.p_value?.toFixed(6)} />
              <ResultRow
                label="Confidence Interval"
                value={
                  freqResult.confidence_interval
                    ? `[${freqResult.confidence_interval[0]?.toFixed(4)}, ${freqResult.confidence_interval[1]?.toFixed(4)}]`
                    : "—"
                }
              />
              <ResultRow
                label="Significant"
                value={
                  <span
                    className={
                      freqResult.significant
                        ? "text-success-400"
                        : "text-warning-400"
                    }
                  >
                    {freqResult.significant ? "Yes ✓" : "No ✗"}
                  </span>
                }
              />
            </tbody>
          </table>
        )}
      </section>

      {/* Bayesian */}
      <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
        <h3 className="text-sm font-medium text-surface-300 mb-4">
          Bayesian Analysis
        </h3>
        <form onSubmit={handleBayes} className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {BAYES_FIELDS.map((f) => (
            <InputField
              key={f.name}
              label={f.label}
              name={f.name}
              value={bayesForm[f.name]}
              onChange={onChangeBayes}
            />
          ))}
          <div className="col-span-full">
            <button
              type="submit"
              disabled={loading.bayes}
              className="px-4 py-2 rounded-md bg-accent-500 text-white text-sm font-medium hover:bg-accent-400 disabled:opacity-50 transition-colors cursor-pointer"
            >
              {loading.bayes ? "Running…" : "Run Bayesian"}
            </button>
          </div>
        </form>

        {error.bayes && (
          <p className="mt-3 text-sm text-danger-400">{error.bayes}</p>
        )}

        {bayesResult && (
          <div className="mt-4 space-y-4">
            <p className="text-sm text-surface-300">
              P(B {">"} A):{" "}
              <span className="font-mono text-accent-300">
                {((bayesResult.prob_b_better ?? bayesResult.probability_b_better) * 100).toFixed(2)}%
              </span>
            </p>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={probBarData}>
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    axisLine={{ stroke: "#334155" }}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    axisLine={{ stroke: "#334155" }}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1e293b",
                      border: "1px solid #334155",
                      borderRadius: 6,
                      color: "#cbd5e1",
                    }}
                    formatter={(v) => (v * 100).toFixed(2) + "%"}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {probBarData.map((_, i) => (
                      <Cell
                        key={i}
                        fill={i === 0 ? "#6366f1" : "#475569"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
