import axios from "axios";
import { QueryClient } from "@tanstack/react-query";
import type {
  FrequentistRequest,
  FrequentistResponse,
  BayesianRequest,
  BayesianResponse,
  SampleSizeRequest,
  SampleSizeResponse,
  CLTRequest,
  CLTResponse,
  PersonaRequest,
  PersonaResponse,
  KMeansRequest,
  KMeansResponse,
  Experiment,
} from "../types";

export const apiClient = axios.create({
  baseURL: "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
});

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 2,
    },
  },
});

// ── A/B Testing ───────────────────────────────────────────────────────────────

export const runFrequentist = (data: FrequentistRequest): Promise<FrequentistResponse> =>
  apiClient.post("/api/v1/test/frequentist", data).then((r) => r.data);

export const calcSampleSize = (data: SampleSizeRequest): Promise<SampleSizeResponse> =>
  apiClient.post("/api/v1/test/sample-size", data).then((r) => r.data);

export const runBayesian = (data: BayesianRequest): Promise<BayesianResponse> =>
  apiClient.post("/api/v1/test/bayesian", data).then((r) => r.data);

// ── Clustering ────────────────────────────────────────────────────────────────

export const discoverPersonas = (data: PersonaRequest): Promise<PersonaResponse> =>
  apiClient.post("/api/v1/clustering/personas", data).then((r) => r.data);

export const runKMeans = (data: KMeansRequest): Promise<KMeansResponse> =>
  apiClient.post("/api/v1/clustering/kmeans", data).then((r) => r.data);

// ── CLT ───────────────────────────────────────────────────────────────────────

export const demonstrateCLT = (data: CLTRequest): Promise<CLTResponse> =>
  apiClient.post("/api/v1/clt/demonstrate", data).then((r) => r.data);

// ── Experiments ───────────────────────────────────────────────────────────────

export const createExperiment = (
  data: Omit<Experiment, "id" | "status" | "created_at" | "updated_at" | "variants">
): Promise<Experiment> =>
  apiClient.post("/api/v1/experiments", data).then((r) => r.data);

export const listExperiments = (): Promise<Experiment[]> =>
  apiClient.get("/api/v1/experiments").then((r) => r.data);

export const getExperiment = (id: string): Promise<Experiment> =>
  apiClient.get(`/api/v1/experiments/${id}`).then((r) => r.data);

export default apiClient;
