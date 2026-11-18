/**
 * PathSnap API client.
 * All backend communication goes through this module.
 * Reads the backend base URL from VITE_API_BASE_URL env var.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const REQUEST_TIMEOUT_MS = 120000; // 2 minutes for inference

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${errText || res.statusText}`);
    }

    if (res.status === 204 || res.headers.get("content-length") === "0") {
      return null;
    }

    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timed out. The backend may be overloaded.");
    }
    throw err;
  } finally {
    clearTimeout(timeoutId);
  }
}

/**
 * Health check.
 */
export async function checkHealth() {
  return request("/health");
}

/**
 * List available trips for a split.
 */
export async function listTrips(split = "student_test") {
  return request(`/api/trips?split=${encodeURIComponent(split)}`);
}

/**
 * Predict route for a trip.
 * Runs candidate generation, classifier scoring, HMM/Viterbi decoding, and route reconstruction.
 */
export async function predictTrip(tripId, split = "student_test", options = {}) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify({
      trip_id: tripId,
      split,
      radius_m: options.radius_m ?? 50,
      max_candidates: options.max_candidates ?? 32,
    }),
  });
}

/**
 * Evaluate a trip against its reference route.
 * Uses reference data for scoring only (never for inference).
 */
export async function evaluateTrip(tripId, split = "student_test", options = {}) {
  return request("/api/evaluate", {
    method: "POST",
    body: JSON.stringify({
      trip_id: tripId,
      split,
      radius_m: options.radius_m ?? 50,
      max_candidates: options.max_candidates ?? 32,
    }),
  });
}

export { BASE_URL };
