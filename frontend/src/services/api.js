/**
 * VoiceGuard AI - API Service Layer
 * ===================================
 * Thin axios wrapper around the FastAPI backend. Centralizing requests here
 * means components never need to know the base URL or response shape
 * details directly.
 */

import axios from "axios";

const api = axios.create({
  baseURL: `${import.meta.env.VITE_API_URL || ""}/api`,
  timeout: 30000,
});

export async function predictVoice(file, onUploadProgress) {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress,
  });
  return data;
}

export async function fetchHistory(limit = 100) {
  const { data } = await api.get("/history", { params: { limit } });
  return data;
}

export async function clearHistory() {
  const { data } = await api.delete("/history");
  return data;
}

export async function fetchHealth() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
