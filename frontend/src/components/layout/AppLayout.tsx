import type { ReactNode } from "react";
import { useAppStore } from "../../store/useAppStore";

type Page = "ab" | "clt" | "personas" | "experiments";

const NAV: { key: Page; label: string; icon: string }[] = [
  { key: "ab", label: "A/B Testing", icon: "⚗️" },
  { key: "clt", label: "CLT Visualizer", icon: "📊" },
  { key: "personas", label: "User Personas", icon: "👥" },
  { key: "experiments", label: "Experiments", icon: "🧪" },
];

interface AppLayoutProps {
  children: ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const { activePage, setActivePage } = useAppStore();

  return (
    <div className="flex h-screen bg-surface-900 text-surface-300 overflow-hidden">
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
                onClick={() => setActivePage(n.key)}
                className={`w-full text-left px-3 py-2 rounded-md text-sm flex items-center gap-2 transition-colors cursor-pointer ${
                  activePage === n.key
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
          v2.0.0
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-auto p-6">{children}</main>
    </div>
  );
}
