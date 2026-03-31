import { useState, useEffect } from "react";
import { createExperiment, listExperiments } from "../api/client";
import type { Experiment } from "../types";
import type { AxiosError } from "axios";

function getErrorMessage(err: unknown): string {
  const axiosErr = err as AxiosError<{ detail: string }>;
  return axiosErr.response?.data?.detail ?? (err instanceof Error ? err.message : String(err));
}

interface ExperimentForm {
  name: string;
  description: string;
  hypothesis: string;
  baseline_rate: number;
}

const EMPTY_FORM: ExperimentForm = {
  name: "",
  description: "",
  hypothesis: "",
  baseline_rate: 0.1,
};

export default function ExperimentList() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [form, setForm] = useState<ExperimentForm>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchExperiments = async () => {
    setFetching(true);
    try {
      const data = await listExperiments();
      setExperiments(Array.isArray(data) ? data : []);
    } catch {
      // silently handle – list may fail if backend is down
    } finally {
      setFetching(false);
    }
  };

  useEffect(() => {
    void fetchExperiments();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await createExperiment({
        name: form.name,
        description: form.description,
        hypothesis: form.hypothesis,
        baseline_rate: Number(form.baseline_rate),
        min_detectable_effect: null,
        confidence_level: 0.95,
      });
      setForm(EMPTY_FORM);
      void fetchExperiments();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <h2 className="text-xl font-semibold text-surface-300">Experiments</h2>

      {/* Create form */}
      <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
        <h3 className="text-sm font-medium text-surface-300 mb-4">New Experiment</h3>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <label className="block">
              <span className="text-xs text-surface-400">Name</span>
              <input
                type="text"
                value={form.name}
                onChange={(e) => setForm((s) => ({ ...s, name: e.target.value }))}
                required
                className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
              />
            </label>
            <label className="block">
              <span className="text-xs text-surface-400">Baseline Rate</span>
              <input
                type="number"
                step="any"
                value={form.baseline_rate}
                onChange={(e) =>
                  setForm((s) => ({ ...s, baseline_rate: Number(e.target.value) }))
                }
                className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
              />
            </label>
          </div>

          <label className="block">
            <span className="text-xs text-surface-400">Hypothesis</span>
            <input
              type="text"
              value={form.hypothesis}
              onChange={(e) => setForm((s) => ({ ...s, hypothesis: e.target.value }))}
              className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500"
            />
          </label>

          <label className="block">
            <span className="text-xs text-surface-400">Description</span>
            <textarea
              value={form.description}
              onChange={(e) => setForm((s) => ({ ...s, description: e.target.value }))}
              rows={2}
              className="mt-1 block w-full rounded-md bg-surface-900 border border-surface-700 px-3 py-1.5 text-sm text-surface-300 focus:outline-none focus:border-accent-500 resize-none"
            />
          </label>

          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 rounded-md bg-accent-500 text-white text-sm font-medium hover:bg-accent-400 disabled:opacity-50 transition-colors cursor-pointer"
          >
            {loading ? "Creating…" : "Create Experiment"}
          </button>

          {error && <p className="text-sm text-danger-400">{error}</p>}
        </form>
      </section>

      {/* List */}
      <section className="bg-surface-800 rounded-lg border border-surface-700 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-surface-300">Existing Experiments</h3>
          <button
            onClick={() => void fetchExperiments()}
            disabled={fetching}
            className="text-xs text-accent-400 hover:text-accent-300 disabled:opacity-50 cursor-pointer"
          >
            {fetching ? "Loading…" : "Refresh"}
          </button>
        </div>

        {experiments.length === 0 ? (
          <p className="text-sm text-surface-500">
            {fetching ? "Loading experiments…" : "No experiments yet."}
          </p>
        ) : (
          <div className="space-y-3">
            {experiments.map((exp, i) => (
              <div
                key={exp.id ?? i}
                className="bg-surface-900 rounded-md p-4 border border-surface-700/50"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="text-sm font-medium text-surface-300">{exp.name}</h4>
                    {exp.hypothesis && (
                      <p className="text-xs text-surface-500 mt-1">{exp.hypothesis}</p>
                    )}
                  </div>
                  {exp.baseline_rate != null && (
                    <span className="text-xs font-mono text-accent-300 bg-accent-500/10 px-2 py-0.5 rounded">
                      {(exp.baseline_rate * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                {exp.description && (
                  <p className="text-xs text-surface-400 mt-2">{exp.description}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
