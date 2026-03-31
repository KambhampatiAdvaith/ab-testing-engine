import ABTestPanel from "./components/ABTestPanel";
import CLTVisualizer from "./components/CLTVisualizer";
import PersonaScatter from "./components/PersonaScatter";
import ExperimentList from "./components/ExperimentList";
import AppLayout from "./components/layout/AppLayout";
import { useAppStore } from "./store/useAppStore";

export default function App() {
  const activePage = useAppStore((s) => s.activePage);

  return (
    <AppLayout>
      {activePage === "ab" && <ABTestPanel />}
      {activePage === "clt" && <CLTVisualizer />}
      {activePage === "personas" && <PersonaScatter />}
      {activePage === "experiments" && <ExperimentList />}
    </AppLayout>
  );
}
