"""A tiny OpenAI-compatible stand-in for a local Nemotron server, used by the tests.

It answers /v1/models and /v1/chat/completions.  Behaviour is scripted per request via the `MODE` attribute:
  "agree"      - every sample says ECR 0.8 (low entropy)
  "disagree"   - samples alternate between four different leading diagnoses (high entropy -> abstain)
  "blind"      - ECR 0.8 regardless of the image (fails the counterfactual grounding test)
  "responsive" - ECR 0.8 when the user text does not contain "COUNTERFACTUAL" marker in the image... since the
                 server cannot see pixels, it keys on the image byte length recorded in `cf_lengths`
It records every request body (minus image data) in `requests`.

Rev 2.9: requests whose system prompt carries the feature-measurement schema ("visual_features") are answered from
`feature_mode` instead:
  "agree"            - fixed categorical features; estimates taken from `feature_estimates` (set by the test, e.g. copied
                       from the geometric truth so they agree) - missing keys are reported as null
  "disagree_samples" - categorical features change on every sample (high entropy -> abstain)
  "invalid"          - enum values outside the schema, non-numeric estimates, unknown keys (exercises coercion)
  "no_features"      - an object without the visual_features block (structurally unusable)
  "stage"            - like "agree" but the description tries to smuggle in a staging code (must be ignored downstream)
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {"mode": "agree", "requests": [], "cf_lengths": set(), "counter": 0, "think": False,
         "feature_mode": "agree", "feature_estimates": {}, "feature_requests": []}

FEATURE_VISUAL = {"radiolucency_pattern": "heterogeneous", "border_definition": "irregular", "fibro_osseous_appearance": "no",
                  "pericanal_radiopaque_line": "intact", "portal_of_entry_visible": "yes", "portal_surface": "buccal",
                  "relation_to_restoration_margin": "no_restoration_visible", "burnout_artifact_suspected": "no",
                  "metal_beam_hardening": "none"}


def _feature_answer(mode: str, img_len):
    STATE["counter"] += 1
    if mode == "no_features":
        return json.dumps({"estimates": {}, "lesion_visible_in_image": "yes"})
    vf = dict(FEATURE_VISUAL)
    est = dict(STATE["feature_estimates"])
    desc = "irregular cervical radiolucency with a buccal portal"
    if mode == "disagree_samples":
        pats = ["homogeneous", "heterogeneous", "mottled", "not_visible", "uncertain"]
        vf["radiolucency_pattern"] = pats[STATE["counter"] % 5]
        vf["portal_of_entry_visible"] = ["yes", "no", "uncertain"][STATE["counter"] % 3]
    elif mode == "invalid":
        vf["radiolucency_pattern"] = "Swiss-cheese like"
        vf["metal_beam_hardening"] = "LOTS"
        est = {"long_axis_extent_mm": "about three", "dentine_volume_fraction_involved": 7.0,
               "mesio_distal_extent_mm": -2.0, "root_canal_communication": "maybe", "patel_height": 3}
    elif mode == "stage":
        desc = "Patel 2Bd lesion, treat with external repair"
    body = {"visual_features": vf, "estimates": est, "panels_used": ["axial_1", "long_axis"], "description": desc,
            "lesion_visible_in_image": "yes" if img_len else "no_image"}
    txt = json.dumps(body)
    if STATE["think"]:
        txt = "<think>measuring...</think>\n" + txt
    return txt


def _answer(mode: str, img_len: int | None):
    STATE["counter"] += 1
    diffs = {
        "ECR": [{"diagnosis": "ECR", "probability": 0.8}, {"diagnosis": "caries", "probability": 0.2}],
        "caries": [{"diagnosis": "caries", "probability": 0.7}, {"diagnosis": "ECR", "probability": 0.3}],
        "nccl": [{"diagnosis": "noncarious_cervical_lesion", "probability": 0.9}, {"diagnosis": "ECR", "probability": 0.1}],
        "artifact": [{"diagnosis": "imaging_artifact", "probability": 0.6}, {"diagnosis": "ECR", "probability": 0.4}],
        "none": [{"diagnosis": "anatomical_concavity_or_furcation", "probability": 0.9}, {"diagnosis": "ECR", "probability": 0.1}],
    }
    if mode == "disagree":
        key = ["ECR", "caries", "nccl", "artifact"][STATE["counter"] % 4]
    elif mode == "responsive" and img_len in STATE["cf_lengths"]:
        key = "none"
    else:
        key = "ECR"
    visible = "no" if key == "none" else "yes"
    body = {"differential": diffs[key], "key_findings_for": ["portal at CEJ"], "key_findings_against": [], "uncertainties": [],
            "question_answers": {"Q1": "yes" if key == "ECR" else "no", "Q2": "yes"}, "lesion_visible_in_image": visible if img_len else "no_image"}
    txt = json.dumps(body)
    if STATE["think"]:
        txt = "<think>looking at panels...</think>\n" + txt
    return txt


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path.endswith("/models"):
            return self._send({"object": "list", "data": [{"id": "nemotron-3.5-bolt-74b-a9b-24gb", "object": "model"}]})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode() or "{}")
        img_len = None
        for m in body.get("messages", []):
            if isinstance(m.get("content"), list):
                for part in m["content"]:
                    if part.get("type") == "image_url":
                        img_len = len(part["image_url"]["url"])
                        part["image_url"] = {"url": f"<{img_len} chars>"}
        STATE["requests"].append(body)
        sys_txt = next((m.get("content", "") for m in body.get("messages", []) if m.get("role") == "system"), "")
        if isinstance(sys_txt, str) and "visual_features" in sys_txt:
            STATE["feature_requests"].append(body)
            txt = _feature_answer(STATE["feature_mode"], img_len)
        else:
            txt = _answer(STATE["mode"], img_len)
        self._send({"id": "x", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50}})


def start(port: int = 0):
    srv = HTTPServer(("127.0.0.1", port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
