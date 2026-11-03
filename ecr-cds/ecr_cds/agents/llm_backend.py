"""Pluggable language/vision-model backends for the reasoning agents.

The agents work without any model (rule-only).  When a backend is supplied it
adds a *second opinion* that is logged next to the rule result; it never
replaces the measurements and cannot by itself raise the confidence band.

Backends
--------
NullBackend              - always returns None (rule-only run)
PerplexitySDKBackend     - text-only structured extraction through `pplx_sdk.llm`
                           (available inside Perplexity Computer sandboxes)
OpenAICompatibleBackend  - any /v1/chat/completions endpoint with JSON output;
                           sends rendered candidate tiles as images when the
                           model is multimodal (Colab: set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL)
LocalHFBackend           - a local transformers model or LoRA adapter produced by
                           ecr_cds.finetune.train_lora (Colab GPU: set LLM_LOCAL_MODEL=<path>)
NemotronLocalBackend     - NVIDIA Nemotron 3.5 Bolt 74B-A9B (24GB) or any local VLM served by
                           llama-server / Ollama / vLLM through the OpenAI-compatible API
                           (set NEMOTRON_BASE_URL, e.g. http://localhost:8080/v1).  Adds
                           discrete-semantic-entropy abstention and optional thinking mode.
LocalDeterministicBackend - no network call.  The feature-measurement path uses it when
                           VLM_BASE_URL is unset: visual features come from deterministic
                           image statistics and physical measurements from voxel geometry.
VisionEndpointBackend    - any OpenAI-compatible vision server at VLM_BASE_URL.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional


class LLMBackend:
    name = "none"
    supports_images = False

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      images_png: Optional[List[bytes]] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class NullBackend(LLMBackend):
    def complete_json(self, system, user, schema, images_png=None):
        return None


class LocalDeterministicBackend(LLMBackend):
    """Marker for the feature path: do not call a model.

    `measure_candidate_features` recognises this class and fills visual features from
    deterministic image statistics.  Physical measurements stay voxel geometry.
    """
    name = "local_deterministic_fallback"
    mode = "local_deterministic_fallback"
    supports_images = False

    def complete_json(self, system, user, schema, images_png=None):
        return None


class PerplexitySDKBackend(LLMBackend):
    name = "perplexity_sdk_text"

    def __init__(self, max_tokens: int = 4000):
        import pplx_sdk  # noqa: F401  (raises ImportError outside the sandbox)
        self._sdk = pplx_sdk
        self.max_tokens = max_tokens

    def complete_json(self, system, user, schema, images_png=None):
        res = self._sdk.llm.extract(items=[user], instruction=system, output_schema=schema,
                                    max_tokens=self.max_tokens)
        r = res[0]
        if r.error:
            return {"_error": str(r.error)}
        return r.result


class OpenAICompatibleBackend(LLMBackend):
    name = "openai_compatible"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, supports_images: bool = True, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or "gpt-4o"
        self.supports_images = supports_images
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("No API key: set LLM_API_KEY (or OPENAI_API_KEY)")

    def complete_json(self, system, user, schema, images_png=None):
        import urllib.request
        content: Any = [{"type": "text", "text": user}]
        if images_png and self.supports_images:
            for png in images_png:
                b64 = base64.b64encode(png).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system + "\nRespond with JSON only, matching this schema: "
                              + json.dumps(schema)},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode())
            txt = out["choices"][0]["message"]["content"]
            return json.loads(txt)
        except Exception as e:   # network / schema failure -> logged, agents fall back to rules
            return {"_error": f"{type(e).__name__}: {e}"}


class VisionEndpointBackend(OpenAICompatibleBackend):
    """OpenAI-compatible vision endpoint selected by VLM_BASE_URL.

    Probes ``/models`` before the conference so a bad URL fails immediately.
    ``VLM_API_KEY`` defaults to ``none`` (local llama.cpp / Ollama / vLLM).
    ``VLM_MODEL`` overrides the first model id the server reports.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = 180.0):
        super().__init__(api_key=api_key or "none", base_url=base_url, model=model or "vlm",
                         supports_images=True, timeout=timeout)
        info = self.probe_server()
        self.server_info = info
        if not model:
            ids = info.get("models") or []
            if not ids:
                raise RuntimeError(f"No model id reported by {self.base_url}/models; set VLM_MODEL")
            self.model = ids[0]
        self.name = f"openai_compatible:{self.model}"
        self.mode = "openai_compatible"

    def probe_server(self) -> Dict[str, Any]:
        import urllib.request
        req = urllib.request.Request(f"{self.base_url}/models",
                                     headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                out = json.loads(resp.read().decode())
            ids = [m.get("id") for m in out.get("data", []) if isinstance(m, dict) and m.get("id")]
            return {"reachable": True, "models": ids, "base_url": self.base_url}
        except Exception as e:
            raise RuntimeError(
                f"VLM server not reachable at {self.base_url} ({type(e).__name__}: {e}). "
                "Check VLM_BASE_URL, or unset it to use the deterministic local fallback."
            ) from e

    def complete_json(self, system, user, schema, images_png=None):
        """One image, JSON object.  Accepts a JSON object wrapped in prose."""
        import urllib.request
        content: Any = [{"type": "text", "text": user}]
        if images_png and self.supports_images:
            png = next((p for p in images_png if p), None)
            if png is not None:
                b64 = base64.b64encode(png).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system + "\nRespond with JSON only, matching this schema: "
                              + json.dumps(schema)},
                             {"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode())
            txt = out["choices"][0]["message"].get("content") or ""
            parsed = _extract_json(txt)
            if parsed is None:
                return {"_error": f"non-JSON output: {txt[:200]!r}"}
            return parsed
        except Exception as e:
            return {"_error": f"{type(e).__name__}: {e}"}


class LocalHFBackend(LLMBackend):
    """Run a (fine-tuned) causal LM in-process with transformers; loads a PEFT adapter directory if present."""
    name = "local_hf"

    def __init__(self, model_path: Optional[str] = None, max_new_tokens: int = 700, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.path = model_path or os.environ.get("LLM_LOCAL_MODEL")
        if not self.path:
            raise ValueError("set LLM_LOCAL_MODEL to a model or adapter directory")
        adapter_cfg = os.path.join(self.path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            from peft import PeftModel
            base = json.load(open(adapter_cfg)).get("base_model_name_or_path")
            self.tok = AutoTokenizer.from_pretrained(self.path)
            model = AutoModelForCausalLM.from_pretrained(base, device_map=device_map, torch_dtype="auto")
            self.model = PeftModel.from_pretrained(model, self.path).eval()
            self.name = f"local_hf:{os.path.basename(self.path.rstrip('/'))}"
        else:
            self.tok = AutoTokenizer.from_pretrained(self.path)
            self.model = AutoModelForCausalLM.from_pretrained(self.path, device_map=device_map, torch_dtype="auto").eval()
        self.max_new_tokens = max_new_tokens
        self._torch = torch

    def complete_json(self, system, user, schema, images_png=None):
        msgs = [{"role": "system", "content": system + "\nRespond with JSON only, matching this schema: " + json.dumps(schema)},
                {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception as e:
            return {"_error": f"non-JSON output: {e}", "_raw": text[:500]}



def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models may emit before the JSON answer."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_thinking(text)
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def default_cluster_key(obj: Dict[str, Any]) -> str:
    """Discrete semantic key for one structured answer: leading diagnosis + the yes/no/uncertain answers.

    Two samples fall in the same semantic cluster when these discrete fields agree.  Using the structured
    fields (not free text) avoids the entailment model that Kim et al. (Eur Radiol 2026) needed for free-form
    answers while keeping the same entropy definition.
    """
    lead = None
    diff = obj.get("differential")
    if diff is None and isinstance(obj.get("visual_features"), dict):
        # rev 2.9 feature-measurement answers: cluster on the categorical visual features
        vf = obj["visual_features"]
        return "features|" + "|".join(f"{k}={str(vf.get(k)).lower()[:24]}" for k in sorted(vf))
    if isinstance(diff, list) and diff:
        try:
            lead = max(diff, key=lambda it: float(it.get("probability", 0) or 0)).get("diagnosis")
        except Exception:
            lead = diff[0].get("diagnosis") if isinstance(diff[0], dict) else None
    ans = []
    q = obj.get("question_answers") or obj.get("diagnostic_questions")
    if isinstance(q, dict):
        for k in sorted(q):
            v = q[k]
            if isinstance(v, dict):
                v = v.get("answer")
            ans.append(f"{k}={str(v).lower()[:12]}")
    return f"{lead}|{'|'.join(ans)}"


def discrete_semantic_entropy(keys: List[str]) -> Dict[str, Any]:
    """Entropy (base 10, as in Kim et al. 2026) of the cluster frequencies; 0 when all samples agree."""
    import math
    from collections import Counter
    n = len(keys)
    cnt = Counter(keys)
    ent = -sum((c / n) * math.log10(c / n) for c in cnt.values()) if n else 0.0
    max_ent = math.log10(n) if n > 1 else 1.0
    modal, modal_n = cnt.most_common(1)[0] if cnt else (None, 0)
    return {"dse": round(ent, 4), "dse_max": round(max_ent, 4), "n_samples": n, "n_clusters": len(cnt),
            "modal_key": modal, "modal_fraction": round(modal_n / n, 3) if n else 0.0}


class NemotronLocalBackend(OpenAICompatibleBackend):
    """Local NVIDIA Nemotron 3.5 Bolt (or any local VLM) behind an OpenAI-compatible server.

    Serving options (all expose /v1/chat/completions):
      llama-server --model <Bolt-24GB.gguf> --mmproj <mmproj.gguf> --jinja ...   -> http://localhost:8080/v1
      ollama run nemotron-3.5-bolt:74b-a9b                                          -> http://localhost:11434/v1
      vllm serve nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16 ...                   -> http://localhost:8000/v1

    Design (see docs/NEMOTRON_LOCAL.md):
      * one evidence frame per request (the vLLM recipe limits image inputs to 1);
      * sampling at the model-card setting (temperature 1.0, top_p 0.95) for `n_samples` draws, then the modal
        structured answer is returned with its discrete semantic entropy; the caller abstains above
        `dse_threshold` (Kim et al. 2026 used 0.3 / 0.6 of a 15-sample log10 scale);
      * `enable_thinking` toggles the chat-template reasoning mode via chat_template_kwargs;
      * the model is a second opinion only - the rule posterior and measurements are never replaced.
    """
    name = "nemotron_local"

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None,
                 n_samples: int = 5, temperature: float = 1.0, top_p: float = 0.95, dse_threshold: float = 0.3,
                 enable_thinking: bool = False, medium_effort: bool = False, supports_images: bool = True,
                 timeout: float = 600.0, max_tokens: int = 1500, json_mode: str = "response_format",
                 cluster_key=None, probe: bool = True):
        self.base_url = (base_url or os.environ.get("NEMOTRON_BASE_URL") or "http://localhost:8080/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("NEMOTRON_API_KEY") or "none"
        self.model = model or os.environ.get("NEMOTRON_MODEL")
        self.supports_images = supports_images
        self.timeout = timeout
        self.n_samples = max(1, int(n_samples))
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.dse_threshold = float(dse_threshold)
        self.enable_thinking = bool(enable_thinking)
        self.medium_effort = bool(medium_effort)
        self.max_tokens = int(max_tokens)
        self.json_mode = json_mode
        self.cluster_key = cluster_key or default_cluster_key
        self.server_info: Dict[str, Any] = {}
        if probe:
            self.server_info = self.probe_server()
            if not self.model:
                ids = self.server_info.get("models") or []
                if not ids:
                    raise RuntimeError(f"No model id reported by {self.base_url}/models; set NEMOTRON_MODEL")
                self.model = ids[0]
        elif not self.model:
            self.model = "nemotron-3.5-bolt"
        self.name = f"nemotron_local:{self.model}"

    # -- server ------------------------------------------------------------------
    def probe_server(self) -> Dict[str, Any]:
        import urllib.request
        req = urllib.request.Request(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                out = json.loads(resp.read().decode())
            ids = [m.get("id") for m in out.get("data", []) if isinstance(m, dict)]
            return {"reachable": True, "models": ids, "base_url": self.base_url}
        except Exception as e:
            raise RuntimeError(f"Nemotron server not reachable at {self.base_url} ({type(e).__name__}: {e}). "
                               f"Start llama-server / ollama / vllm first or set NEMOTRON_BASE_URL.") from e

    @property
    def available(self) -> bool:
        return bool(self.server_info.get("reachable"))

    def _one(self, system: str, user: str, schema: Dict[str, Any], images_png: Optional[List[bytes]], temperature: float):
        import urllib.request
        content: Any = [{"type": "text", "text": user}]
        if images_png and self.supports_images:
            png = next((p for p in images_png if p), None)   # single image per request
            if png is not None:
                b64 = base64.b64encode(png).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        body: Dict[str, Any] = {"model": self.model, "temperature": temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
                                "messages": [{"role": "system", "content": system + "\nRespond with JSON only, matching this schema: "
                                              + json.dumps(schema)},
                                             {"role": "user", "content": content}],
                                "chat_template_kwargs": {"enable_thinking": self.enable_thinking,
                                                         **({"medium_effort": True} if self.medium_effort else {})}}
        if self.json_mode == "response_format":
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode())
        msg = out["choices"][0]["message"]
        txt = msg.get("content") or ""
        parsed = _extract_json(txt)
        if parsed is None:
            raise ValueError(f"non-JSON output: {txt[:200]!r}")
        usage = out.get("usage") or {}
        return parsed, usage

    def complete_json(self, system, user, schema, images_png=None):
        samples, usage_total, errors = [], {"prompt_tokens": 0, "completion_tokens": 0}, []
        for i in range(self.n_samples):
            temp = self.temperature if self.n_samples > 1 else min(self.temperature, 0.1)
            try:
                obj, usage = self._one(system, user, schema, images_png, temp)
                samples.append(obj)
                for k in usage_total:
                    usage_total[k] += int(usage.get(k, 0) or 0)
            except Exception as e:
                errors.append(f"{type(e).__name__}: {str(e)[:160]}")
        if not samples:
            return {"_error": "all samples failed: " + " | ".join(errors[:3])}
        keys = [self.cluster_key(o) for o in samples]
        dse = discrete_semantic_entropy(keys)
        modal = next(o for o, k in zip(samples, keys) if k == dse["modal_key"])
        modal = dict(modal)
        modal["_dse"] = dse
        modal["_abstain"] = bool(self.n_samples > 1 and dse["dse"] > self.dse_threshold)
        modal["_dse_threshold"] = self.dse_threshold
        modal["_backend"] = {"name": self.name, "n_samples": self.n_samples, "temperature": self.temperature, "top_p": self.top_p,
                             "enable_thinking": self.enable_thinking, "images_sent": int(bool(images_png and self.supports_images)),
                             "usage": usage_total, "sample_errors": errors}
        return modal


def resolve_feature_backend():
    """Feature-measurement backend for the CBCT path.

    Returns ``(backend, mode_id, detail)``.

    * ``VLM_BASE_URL`` unset → :class:`LocalDeterministicBackend` (no GPU, no network).
    * ``VLM_BASE_URL`` set → :class:`VisionEndpointBackend` (OpenAI-compatible chat completions).
      Optional ``VLM_API_KEY``, ``VLM_MODEL``, ``VLM_TIMEOUT``.

    A set but unreachable URL raises ``RuntimeError``.  It does not silently fall back.
    """
    url = (os.environ.get("VLM_BASE_URL") or "").strip()
    if not url:
        backend = LocalDeterministicBackend()
        return backend, "local_deterministic_fallback", "VLM_BASE_URL unset"
    timeout = float(os.environ.get("VLM_TIMEOUT") or 180)
    backend = VisionEndpointBackend(base_url=url, api_key=os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                                    model=(os.environ.get("VLM_MODEL") or "").strip() or None, timeout=timeout)
    return backend, "openai_compatible", f"VLM_BASE_URL={url} model={backend.model}"


def auto_backend(prefer: str = "auto") -> LLMBackend:
    """Pick a backend from the environment without failing: rule-only if nothing is configured.
    Order: local Nemotron server (NEMOTRON_BASE_URL) > local fine-tuned model (LLM_LOCAL_MODEL)
           > OpenAI-compatible endpoint (LLM_API_KEY) > Perplexity SDK."""
    if prefer == "none":
        return NullBackend()
    if prefer in ("auto", "nemotron") and (prefer == "nemotron" or os.environ.get("NEMOTRON_BASE_URL")):
        try:
            return NemotronLocalBackend()
        except Exception:
            if prefer == "nemotron":
                raise
    if prefer in ("auto", "local") and os.environ.get("LLM_LOCAL_MODEL"):
        try:
            return LocalHFBackend()
        except Exception:
            pass
    if prefer in ("auto", "openai") and (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        try:
            return OpenAICompatibleBackend()
        except Exception:
            pass
    if prefer in ("auto", "perplexity"):
        try:
            return PerplexitySDKBackend()
        except Exception:
            pass
    return NullBackend()
