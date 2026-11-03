"""A tiny OpenAI-compatible stand-in for a local Nemotron server, used by the tests.

It answers /v1/models and /v1/chat/completions.  Behaviour is scripted per request via the `MODE` attribute:
  "agree"      - every sample says ECR 0.8 (low entropy)
  "disagree"   - samples alternate between four different leading diagnoses (high entropy -> abstain)
  "blind"      - ECR 0.8 regardless of the image (fails the counterfactual grounding test)
  "responsive" - ECR 0.8 when the user text does not contain "COUNTERFACTUAL" marker in the image... since the
                 server cannot see pixels, it keys on the image byte length recorded in `cf_lengths`
It records every request body (minus image data) in `requests`.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {"mode": "agree", "requests": [], "cf_lengths": set(), "counter": 0, "think": False}


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
        txt = _answer(STATE["mode"], img_len)
        self._send({"id": "x", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50}})


def start(port: int = 0):
    srv = HTTPServer(("127.0.0.1", port), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
