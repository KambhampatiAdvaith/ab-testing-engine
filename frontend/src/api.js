import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  headers: { "Content-Type": "application/json" },
});

export const runFrequentist = (data) =>
  api.post("/api/v1/test/frequentist", data).then((r) => r.data);

export const calcSampleSize = (data) =>
  api.post("/api/v1/test/sample-size", data).then((r) => r.data);

export const runBayesian = (data) =>
  api.post("/api/v1/test/bayesian", data).then((r) => r.data);

export const discoverPersonas = (data) =>
  api.post("/api/v1/clustering/personas", data).then((r) => r.data);

export const runKMeans = (data) =>
  api.post("/api/v1/clustering/kmeans", data).then((r) => r.data);

export const demonstrateCLT = (data) =>
  api.post("/api/v1/clt/demonstrate", data).then((r) => r.data);

export const createExperiment = (data) =>
  api.post("/api/v1/experiments", data).then((r) => r.data);

export const listExperiments = () =>
  api.get("/api/v1/experiments").then((r) => r.data);

export default api;
