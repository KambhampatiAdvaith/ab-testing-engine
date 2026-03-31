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
import { runFrequentist, runBayesian } from "../api/client";
import type { FrequentistResponse, BayesianResponse } from "../types";
import type { AxiosError } from "axios";

interface FieldDef {
  label: string;
  name: string;
  defaultVal: number;
}

const FREQ_FIELDS: FieldDef[] = [
  { label: "Control Clicks", name: "control_clicks", defaultVal: 120 },
  { label: "Control Total", name: "control_total", defaultVal: 1000 },
  { label: "Variant Clicks", name: "variant_clicks", defaultVal: 145 },
  { label: "Variant Total", name: "variant_total", defaultVal: 1000 },
  { label: "Alpha (α)", name: "alpha", defaultVal: 0.05 },
];

const BAYES_FIELDS: FieldDef[] = [
  { label: "Control Successes", name: "control_successes", defaultVal: 120 },
  { label: "Control Failures", name: "control_failures", defaultVal: 880 },
  { label: "Variant Successes", name: "variant_successes", defaultVal: 145 },
  { label: "Variant Failures", name: "variant_failures", defaultVal: 855 },
  { label: "Prior α", name: "prior_alpha", defaultVal: 1 },
  { label: "Prior β", name: "prior_beta", defaultVal: 1 },
  { label: "Simulations", name: "n_simulations", defaultVal: 10000 },
];

interface InputFieldProps {
  label: string;
  name: string;
  value: number | string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

function InputField({ label, name, value, onChange }: InputFieldProps) {
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

interface ResultRowProps {
  label: string;
  value: React.ReactNode;
}

function ResultRow({ label, value }: ResultRowProps) {
  return (
    <tr className="border-b border-surface-700/50">
      <td className="py-2 pr-4 text-sm text-surface-400">{label}</td>
      <td className="py-2 text-sm text-surface-300 font-mono">{value}</td>
    </tr>
  );
}

function getErrorMessage(err: unknown): string {
  const axiosErr = err as AxiosError<{ detail: string }>;
  return axiosErr.response?.data?.detail ?? (err instanceof Error ? err.message : String(err));
}

export default function ABTestPanel() {
  const [freqForm, setFreqForm] = useState<Record<string, number | string>>(
    Object.fromEntries(FREQ_FIELDS.map((f) => [f.name, f.defaultVal]))
  );
  const [bayesForm, setBayesForm] = useState<Record<string, number | string>>(
    Object.fromEntries(BAYES_FIELDS.map((f) => [f.name, f.defaultVal]))
  );
  const [freqResult, setFreqResult] = useState<FrequentistResponse | null>(null);
  const [bayesResult, setBayesResult] = useState<BayesianResponse | null>(null);
  const [loading, setLoading] = useState({ freq: false, bayes: false });
  const [error, setError] = useState<{ freq: string | null; bayes: string | null }>({
    freq: null,
    bayes: null,
  });

  const handleFreq = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading((s) => ({ ...s, freq: true }));
    setError((s) => ({ ...s, freq: null }));
    try {
      const parsed = Object.fromEntries(
        Object.entries(freqForm).map(([k, v]) => [k, Number(v)])
      ) as unknown as Parameters<typeof runFrequentist>[0];
      setFreqResult(await runFrequentist(parsed));
    } catch (err) {
      setError((s) => ({ ...s, freq: getErrorMessage(err) }));
    } finally {
      setLoading((s) => ({ ...s, freq: false }));
    }
  };

  const handleBayes = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading((s) => ({ ...s, bayes: true }));
    setError((s) => ({ ...s, bayes: null }));
    try {
      const parsed = Object.fromEntries(
        Object.entries(bayesForm).map(([k, v]) => [k, Number(v)])
      ) as unknown as Parameters<typeof runBayesian>[0];
      setBayesResult(await runBayesian(parsed));
    } catch (err) {
      setError((s) => ({ ...s, bayes: getErrorMessage(err) }));
    } finally {
      setLoading((s) => ({ ...s, bayes: false }));
    }
  };

  const onChangeFreq = (e: React.ChangeEvent<HTMLInputElement>) =>
    setFreqForm((s) => ({ ...s, [e.target.name]: e.target.value }));
  const onChangeBayes = (e: React.ChangeEvent<HTMLInputElement>) =>
    setBayesForm((s) => ({ ...s, [e.target.name]: e.target.value }));

  const probBBeatsA = bayesResult?.probability_b_beats_a ?? 0;
  const probBarData = bayesResult
    ? [
        { name: "P(B > A)", value: probBBeatsA },
        { name: "P(A ≥ B)", value: 1 - probBBeatsA },
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

        {loading.freq && !freqResult && (
          <div className="mt-4 space-y-2 animate-pulse">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-8 bg-surface-700 rounded" />
            ))}
          </div>
        )}

        {freqResult && (
          <table className="mt-4 w-full">
            <tbody>
              <ResultRow label="Z-Statistic" value={freqResult.z_statistic?.toFixed(4)} />
              <ResultRow label="P-Value" value={freqResult.p_value?.toFixed(6)} />
              <ResultRow
                label="Confidence Interval"
                value={`[${freqResult.confidence_interval[0]?.toFixed(4)}, ${freqResult.confidence_interval[1]?.toFixed(4)}]`}
              />
              <ResultRow
                label="Significant"
                value={
                  <span className={freqResult.is_significant ? "text-green-400" : "text-yellow-400"}>
                    {freqResult.is_significant ? "Yes ✓" : "No ✗"}
                  </span>
                }
              />
              <ResultRow label="Control Rate" value={freqResult.control_rate?.toFixed(4)} />
              <ResultRow label="Variant Rate" value={freqResult.variant_rate?.toFixed(4)} />
              <ResultRow
                label="Relative Uplift"
                value={`${(freqResult.relative_uplift * 100)?.toFixed(2)}%`}
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

        {loading.bayes && !bayesResult && (
          <div className="mt-4 h-48 bg-surface-700 rounded animate-pulse" />
        )}

        {bayesResult && (
          <div className="mt-4 space-y-4">
            <p className="text-sm text-surface-300">
              P(B &gt; A):{" "}
              <span className="font-mono text-accent-300">
                {(probBBeatsA * 100).toFixed(2)}%
              </span>
              {bayesResult.recommendation && (
                <span className="ml-3 text-xs text-surface-400">
                  Recommendation: {bayesResult.recommendation}
                </span>
              )}
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
                    formatter={(v) => `${((v as number) * 100).toFixed(2)}%`}
                  />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {probBarData.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? "#6366f1" : "#475569"} />
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
