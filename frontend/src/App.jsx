import { useState } from "react";
import ABTestPanel from "./components/ABTestPanel";
import CLTVisualizer from "./components/CLTVisualizer";
import PersonaScatter from "./components/PersonaScatter";
import ExperimentList from "./components/ExperimentList";

const NAV = [
  { key: "ab", label: "A/B Testing", icon: "⚗️" },
  { key: "clt", label: "CLT Visualizer", icon: "📊" },
  { key: "personas", label: "User Personas", icon: "👥" },
  { key: "experiments", label: "Experiments", icon: "🧪" },
];

export default function App() {
  const [page, setPage] = useState("ab");

  return (
    <>
      {/* Sidebar */}
      <nav className="w-56 shrink-0 bg-surface-800 border-r border-surface-700 flex flex-col">
        <div className="px-4 py-5 border-b border-surface-700">
          <h1 className="text-sm font-semibold tracking-wider text-surface-300 uppercase">
            A/B Engine
          </h1>
        </div>
        <ul className="flex-1 py-3 space-y-1 px-2">
          {NAV.map((n) => (
            <li key={n.key}>
              <button
                onClick={() => setPage(n.key)}
                className={`w-full text-left px-3 py-2 rounded-md text-sm flex items-center gap-2 transition-colors cursor-pointer ${
                  page === n.key
                    ? "bg-accent-500/20 text-accent-300"
                    : "text-surface-400 hover:bg-surface-700 hover:text-surface-300"
                }`}
              >
                <span>{n.icon}</span>
                {n.label}
              </button>
            </li>
          ))}
        </ul>
        <div className="px-4 py-3 border-t border-surface-700 text-xs text-surface-500">
          v1.0.0
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">
        {page === "ab" && <ABTestPanel />}
        {page === "clt" && <CLTVisualizer />}
        {page === "personas" && <PersonaScatter />}
        {page === "experiments" && <ExperimentList />}
      </main>
    </>
  );
}
