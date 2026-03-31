import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { demonstrateCLT } from "../api/client";
import type { CLTResponse } from "../types";
import type { AxiosError } from "axios";

const DISTRIBUTIONS = ["normal", "exponential", "uniform", "binomial"] as const;
const DEFAULT_SAMPLE_SIZES = "5,20,50,100";
const COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#818cf8"];

function getErrorMessage(err: unknown): string {
  const axiosErr = err as AxiosError<{ detail: string }>;
  return axiosErr.response?.data?.detail ?? (err instanceof Error ? err.message : String(err));
}

interface HistogramBin {
  bin: string;
  count: number;
}

function buildHistogram(means: number[]): HistogramBin[] {
  if (!means || means.length === 0) return [];
  const min = Math.min(...means);
  const max = Math.max(...means);
  const bins = 30;
  const width = (max - min) / bins || 1;
  const counts = Array<number>(bins).fill(0);
  for (const m of means) {
    const idx = Math.min(Math.floor((m - min) / width), bins - 1);
    counts[idx]++;
  }
  return counts.map((count, i) => ({
    bin: (min + (i + 0.5) * width).toFixed(2),
    count,
  }));
}

export default function CLTVisualizer() {
  const [form, setForm] = useState({
    distribution: "exponential" as (typeof DISTRIBUTIONS)[number],
    sample_sizes: DEFAULT_SAMPLE_SIZES,
    n_simulations: 1000,
  });
  const [result, setResult] = useState<CLTResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSize, setActiveSize] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = {
        distribution: form.distribution,
        sample_sizes: form.sample_sizes
          .split(",")
          .map((s) => parseInt(s.trim(), 10))
          .filter((n) => !isNaN(n)),
        n_simulations: Number(form.n_simulations),
        params: {},
      };
      const res = await demonstrateCLT(payload);
      setResult(res);
      setActiveSize(String(payload.sample_sizes[0]));
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const resultsMap: Record<string, CLTResponse["results"][number]> = {};
  if (result?.results) {
    for (const entry of result.results) {
      resultsMap[String(entry.sample_size)] = entry;
    }
  }
  const sampleSizeKeys = Object.keys(resultsMap);

  const activeMeans =
    activeSize != null && resultsMap[activeSize]
      ? resultsMap[activeSize].sample_means ?? []
      : [];

  const chartData = buildHistogram(activeMeans);

  return (
    <div className="space-y-6 max-w-4xl">
      <h2 className="text-xl font-semibold text-surface-300">
        Central Limit Theorem Visualizer
      </h2>

      <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
        <form onSubmit={handleSubmit} className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          <label className="block">
            <span className="text-xs text-surface-400">Distribution</span>
            <select
              value={form.distribution}
              onChange={(e) =>
                setForm((s) => ({
                  ...s,
                  distribution: e.target.value as (typeof DISTRIBUTIONS)[number],
                }))
              }
              className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
            >
              {DISTRIBUTIONS.map((d) => (
                <option key={d} value={d}>
                  {d.charAt(0).toUpperCase() + d.slice(1)}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="text-xs text-surface-400">
              Sample Sizes (comma-separated)
            </span>
            <input
              type="text"
              value={form.sample_sizes}
              onChange={(e) =>
                setForm((s) => ({ ...s, sample_sizes: e.target.value }))
              }
              className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
            />
          </label>

          <label className="block">
            <span className="text-xs text-surface-400">Simulations</span>
            <input
              type="number"
              value={form.n_simulations}
              onChange={(e) =>
                setForm((s) => ({ ...s, n_simulations: Number(e.target.value) }))
              }
              className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
            />
          </label>

          <div className="col-span-full">
            <button
              type="submit"
              disabled={loading}
              className="px-4 py-2 rounded-md bg-accent-500 text-white text-sm font-medium hover:bg-accent-400 disabled:opacity-50 transition-colors cursor-pointer"
            >
              {loading ? "Simulating…" : "Run Simulation"}
            </button>
          </div>
        </form>

        {error && <p className="mt-3 text-sm text-danger-400">{error}</p>}
      </section>

      {loading && !result && (
        <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
          <div className="h-72 bg-surface-700 rounded animate-pulse" />
        </section>
      )}

      {result && sampleSizeKeys.length > 0 && (
        <section className="bg-surface-800 rounded-lg border border-surface-700 p-5 space-y-4">
          <div className="flex gap-2 flex-wrap">
            {sampleSizeKeys.map((key, i) => (
              <button
                key={key}
                onClick={() => setActiveSize(key)}
                className={`px-3 py-1 rounded text-xs font-medium transition-colors cursor-pointer ${
                  String(activeSize) === String(key)
                    ? "text-white"
                    : "bg-surface-700 text-surface-400 hover:text-surface-300"
                }`}
                style={
                  String(activeSize) === String(key)
                    ? { backgroundColor: COLORS[i % COLORS.length] }
                    : {}
                }
              >
                n = {key}
              </button>
            ))}
          </div>

          {chartData.length > 0 ? (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <XAxis
                    dataKey="bin"
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                    axisLine={{ stroke: "#334155" }}
                    tickLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
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
                  />
                  <Bar
                    dataKey="count"
                    fill={
                      COLORS[
                        sampleSizeKeys.indexOf(String(activeSize)) % COLORS.length
                      ]
                    }
                    radius={[2, 2, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-surface-500">No data for this sample size.</p>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {sampleSizeKeys.map((key) => {
              const entry = resultsMap[key];
              return (
                <div key={key} className="bg-surface-900 rounded-md p-3 text-center">
                  <p className="text-xs text-surface-500 mb-1">n = {key}</p>
                  <p className="text-sm font-mono text-surface-300">
                    μ = {entry?.empirical_mean?.toFixed(4) ?? "—"}
                  </p>
                  <p className="text-sm font-mono text-surface-300">
                    σ = {entry?.empirical_std?.toFixed(4) ?? "—"}
                  </p>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
