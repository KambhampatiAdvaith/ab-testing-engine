import { useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { discoverPersonas } from "../api";

const CLUSTER_COLORS = [
  "#6366f1",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#818cf8",
  "#34d399",
  "#fbbf24",
  "#f87171",
];

export default function PersonaScatter() {
  const [form, setForm] = useState({ n_users: 200, k: 4 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await discoverPersonas({
        n_users: Number(form.n_users),
        k: Number(form.k),
      });
      setResult(res);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  // Build per-cluster data arrays for ScatterChart
  const clusters = {};
  if (result?.users) {
    for (const u of result.users) {
      const c = u.cluster ?? u.persona ?? 0;
      if (!clusters[c]) clusters[c] = [];
      clusters[c].push({
        x: u.session_duration ?? u.feature1 ?? u.x ?? 0,
        y: u.pages_viewed ?? u.feature2 ?? u.y ?? 0,
        name: u.user_id ?? u.id,
      });
    }
  }

  const clusterKeys = Object.keys(clusters).sort(
    (a, b) => Number(a) - Number(b)
  );

  return (
    <div className="space-y-6 max-w-4xl">
      <h2 className="text-xl font-semibold text-surface-300">User Personas</h2>

      <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
        <form onSubmit={handleSubmit} className="flex items-end gap-4 flex-wrap">
          <label className="block">
            <span className="text-xs text-surface-400">Number of Users</span>
            <input
              type="number"
              value={form.n_users}
              onChange={(e) =>
                setForm((s) => ({ ...s, n_users: e.target.value }))
              }
              className="mt-1 block w-36 rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
            />
          </label>

          <label className="block">
            <span className="text-xs text-surface-400">Clusters (k)</span>
            <input
              type="number"
              value={form.k}
              onChange={(e) =>
                setForm((s) => ({ ...s, k: e.target.value }))
              }
              className="mt-1 block w-28 rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
            />
          </label>

          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 rounded-md bg-accent-500 text-white text-sm font-medium hover:bg-accent-400 disabled:opacity-50 transition-colors cursor-pointer"
          >
            {loading ? "Discovering…" : "Discover Personas"}
          </button>
        </form>

        {error && <p className="mt-3 text-sm text-danger-400">{error}</p>}
      </section>

      {result && clusterKeys.length > 0 && (
        <section className="bg-surface-800 rounded-lg border border-surface-700 p-5 space-y-4">
          <h3 className="text-sm font-medium text-surface-300">
            Cluster Scatter Plot
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Feature 1"
                  tick={{ fill: "#94a3b8", fontSize: 12 }}
                  axisLine={{ stroke: "#334155" }}
                  tickLine={false}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Feature 2"
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
                  cursor={{ strokeDasharray: "3 3", stroke: "#475569" }}
                />
                <Legend
                  wrapperStyle={{ color: "#94a3b8", fontSize: 12 }}
                />
                {clusterKeys.map((key, i) => (
                  <Scatter
                    key={key}
                    name={`Cluster ${key}`}
                    data={clusters[key]}
                    fill={CLUSTER_COLORS[i % CLUSTER_COLORS.length]}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {result?.personas && (
        <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
          <h3 className="text-sm font-medium text-surface-300 mb-3">
            Persona Analysis
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700 text-surface-400">
                  <th className="text-left py-2 pr-4 font-medium">Persona</th>
                  <th className="text-left py-2 pr-4 font-medium">Size</th>
                  <th className="text-left py-2 pr-4 font-medium">
                    Avg Session Duration
                  </th>
                  <th className="text-left py-2 font-medium">
                    Avg Pages Viewed
                  </th>
                </tr>
              </thead>
              <tbody>
                {result.personas.map((p, i) => (
                  <tr key={i} className="border-b border-surface-700/50">
                    <td className="py-2 pr-4 font-mono text-accent-300">
                      {p.name ?? `Cluster ${p.cluster ?? i}`}
                    </td>
                    <td className="py-2 pr-4 font-mono text-surface-300">
                      {p.size ?? p.count ?? "—"}
                    </td>
                    <td className="py-2 pr-4 font-mono text-surface-300">
                      {p.avg_session_duration?.toFixed(1) ??
                        p.centroid?.[0]?.toFixed(1) ??
                        "—"}
                    </td>
                    <td className="py-2 font-mono text-surface-300">
                      {p.avg_pages_viewed?.toFixed(1) ??
                        p.centroid?.[1]?.toFixed(1) ??
                        "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {result && clusterKeys.length === 0 && !result.personas && (
        <p className="text-sm text-surface-500">
          No cluster data returned. Check backend response format.
        </p>
      )}
    </div>
  );
}
