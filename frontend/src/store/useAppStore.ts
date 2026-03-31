import { create } from "zustand";

type Page = "ab" | "clt" | "personas" | "experiments";

interface AppState {
  activePage: Page;
  selectedExperimentId: string | null;
  darkMode: boolean;
  setActivePage: (page: Page) => void;
  setSelectedExperimentId: (id: string | null) => void;
  toggleDarkMode: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  activePage: "ab",
  selectedExperimentId: null,
  darkMode: true,
  setActivePage: (page) => set({ activePage: page }),
  setSelectedExperimentId: (id) => set({ selectedExperimentId: id }),
  toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),
}));
