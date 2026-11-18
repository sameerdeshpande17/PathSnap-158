# Initial backend/frontend contract — v0.1 (design, not running)

Laptop 1 owns parsing, candidates, features, model, HMM and evaluation.
Laptop 2 owns the UI and sends only selected trip IDs. JSON UTF-8 over HTTP.
Proposed base URL: `http://<laptop-1-LAN-IP>:8000/api/v1`.

## Endpoints

### `GET /health`
200: `{"status":"ready","pipeline_version":"<artifact hash>","mode":"evaluation"}`
503 when required artifacts are unavailable. Do not advertise ready with no model.

### `GET /trips`
200: `{"trips":[{"trip_id":"<opaque ID>","observation_count":123}]}`
In final evaluation, list exactly the ten server-configured evaluator trips.
Development deployment uses a separate explicit student-testing allowlist.
Counts above are schema examples, not measured results.
Do not return route IDs, reference geometry, reference labels or filesystem paths.

### `POST /match`
Request: `{"trip_id":"<selected ID>","decoder":"hmm"}`
`decoder` enum: `classifier`, `hmm`; later a separately labelled
`geometric_baseline` development mode may be exposed.

200 response fields:

| Field | Type | Meaning |
|---|---|---|
| request_id | string | Traceable request identifier |
| trip_id | string | Requested allowlisted trip |
| pipeline_version | string | Artifact/configuration identity |
| method | string | Actual method executed |
| observations | GeoJSON FeatureCollection | GPS Point geometry `[longitude, latitude]`; observation index and timestamp_s |
| road_network | GeoJSON FeatureCollection | Relevant LineStrings; local arc_id and directed endpoint IDs |
| assignments | array | observation_index; nullable arc_id; nullable probability; status |
| predicted_route | GeoJSON FeatureCollection | Directed LineStrings in traversal order; split discontinuities explicitly |
| metrics | object | nullable Top-1/F1/log-loss; continuity value and definition; label policy; unavailable reasons |
| timing_ms | object | candidates/features/model/decoder/evaluation/total |
| warnings | array of strings | Gaps, missing candidates, disconnected route portions |

Do not present baseline distances or reference membership as accuracy. Do not
return a numeric probability for the geometric baseline. Null arc IDs indicate
unmatched observations; never silently omit them. A baseline arc union must not
be presented as a reconstructed path.

Errors: 400 invalid JSON; 404 unknown/nonallowlisted trip; 422 invalid options;
503 unloaded model; 500 unexpected failure with request_id, no stack trace.
Trip IDs are keys in an allowlist, never user-controlled filesystem paths.

## Isolation, logging and integration acceptance

Inference loader accepts only track/nodes/arcs. A separate scoring service may
load route data after predictions exist. No truth field is accepted in requests
or returned in responses. Metric names and label provenance accompany scoring.
Evaluator data must not be loaded in development mode.

Later start command (requires implementation):
`~/ml312/bin/python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000`

Configure exact frontend origins through `PATHSNAP_CORS_ORIGINS`, validate LAN
reachability from laptop 2, and avoid hardcoded laptop IPs in frontend code.
Log request ID, trip ID, artifact version, status and timings on laptop 1.
Final acceptance: all ten trips selectable, requests visible in server logs,
matching route and metrics rendered in UI, no reference data in browser payload,
clear handling of failures, and screenshots from both laptops.

API, CORS, network integration and acceptance tests are not yet implemented.
