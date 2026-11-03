"""Shared evidence store ("blackboard") for the multi-agent case conference.

Every agent reads the blackboard and appends *Evidence* items with explicit
provenance.  Nothing is overwritten: a later agent that disagrees adds a
counter-finding, and the moderator resolves the two.  The complete trail is
serialised into the case report so a clinician can audit every step.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    agent: str                      # which agent produced it
    kind: str                       # e.g. "qa", "anatomy", "candidate", "differential", "challenge", "patel", "plan"
    content: Dict[str, Any]         # structured payload (JSON-serialisable)
    confidence: Optional[float] = None   # 0..1, *uncalibrated* unless stated in `basis`
    basis: str = "rule"             # "rule" | "measurement" | "llm" | "vlm" | "clinician"
    caveats: List[str] = field(default_factory=list)
    refers_to: Optional[int] = None  # candidate component id, when applicable
    round: int = 0
    t: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["t"] = round(self.t, 3)
        return d


class Blackboard:
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.items: List[Evidence] = []
        self.arrays: Dict[str, Any] = {}      # large numpy objects, never serialised
        self.transcript: List[Dict[str, Any]] = []   # agent messages in order

    # -- writing --------------------------------------------------------
    def post(self, ev: Evidence) -> Evidence:
        self.items.append(ev)
        return ev

    def say(self, agent: str, text: str, round: int = 0) -> None:
        self.transcript.append({"agent": agent, "round": round, "text": text})

    # -- reading --------------------------------------------------------
    def query(self, kind: Optional[str] = None, agent: Optional[str] = None,
              refers_to: Optional[int] = None) -> List[Evidence]:
        out = self.items
        if kind is not None:
            out = [e for e in out if e.kind == kind]
        if agent is not None:
            out = [e for e in out if e.agent == agent]
        if refers_to is not None:
            out = [e for e in out if e.refers_to == refers_to]
        return list(out)

    def latest(self, kind: str, refers_to: Optional[int] = None) -> Optional[Evidence]:
        q = self.query(kind=kind, refers_to=refers_to)
        return q[-1] if q else None

    def to_dict(self) -> Dict[str, Any]:
        return {"case_id": self.case_id, "evidence": [e.to_dict() for e in self.items],
                "transcript": self.transcript}

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
