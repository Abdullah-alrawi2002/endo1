"""Guideline knowledge base used by the conference's KnowledgeAgent.

Statements are short, cited paraphrases of the ESE 2023 position statement, Patel 2018 (classification and
management), the DMFR 2023 mimics review, the AAE CDT 2024 code guide and the Diagnocat documentation.
They justify each diagnostic question and each plan step and map treatments to CDT codes.  They are
reference text for clinicians, not a substitute for reading the sources.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

from ..decision_matrix import Treatment

_PATH = os.path.join(os.path.dirname(__file__), "guidelines.json")

# Treatment -> (code template by tooth type, descriptor).  "{t}" is replaced by the tooth-type digit:
# anterior 1 / premolar 2 / molar 3 for D347x and D350x, and by D3310/D3320/D3330 for endodontic therapy.
_CDT: Dict[str, List[Dict[str, str]]] = {
    Treatment.MONITOR.value: [{"code": "D0364", "descriptor": "cone beam CT capture and interpretation with limited field of view (review imaging)"},
                              {"code": "D9310", "descriptor": "consultation - diagnostic service"}],
    Treatment.EXTERNAL_REPAIR.value: [{"code": "D347{t}", "descriptor": "surgical repair of root resorption"}],
    Treatment.EXTERNAL_REPAIR_VPT.value: [{"code": "D347{t}", "descriptor": "surgical repair of root resorption"},
                                          {"code": "D3110", "descriptor": "pulp cap - direct (vital pulp therapy; D3220 if pulpotomy)"}],
    Treatment.EXTERNAL_REPAIR_RCT.value: [{"code": "D347{t}", "descriptor": "surgical repair of root resorption"},
                                          {"code": "D33{e}0", "descriptor": "endodontic therapy (excluding final restoration)"}],
    Treatment.INTERNAL_REPAIR_RCT.value: [{"code": "D33{e}0", "descriptor": "endodontic therapy (excluding final restoration)"},
                                          {"code": "D3333", "descriptor": "internal root repair of perforation defects"}],
    Treatment.INTENTIONAL_REPLANT.value: [{"code": "D3470", "descriptor": "intentional reimplantation (including necessary splinting)"}],
    Treatment.HEMISECTION.value: [{"code": "D3920", "descriptor": "hemisection (including any root removal), not including root canal therapy"},
                                  {"code": "D3450", "descriptor": "root amputation - per root (alternative)"}],
    Treatment.DECORONATION.value: [{"code": "D7251", "descriptor": "coronectomy (closest CDT descriptor; decoronation has no dedicated code - confirm with payer)"}],
    Treatment.EXTRACTION.value: [{"code": "D7140", "descriptor": "extraction, erupted tooth or exposed root"},
                                 {"code": "D7210", "descriptor": "extraction requiring removal of bone and/or sectioning (if surgical)"}],
}
_TOOTH_DIGIT = {"anterior": "1", "premolar": "2", "molar": "3"}


def tooth_type_from_iso(iso: Optional[int]) -> str:
    """FDI/ISO two-digit tooth number -> anterior | premolar | molar | unknown."""
    if iso is None:
        return "unknown"
    pos = int(iso) % 10
    if pos in (1, 2, 3):
        return "anterior"
    if pos in (4, 5):
        return "premolar"
    if pos in (6, 7, 8):
        return "molar"
    return "unknown"


def cdt_codes_for(treatment: str, tooth_type: str = "unknown") -> List[Dict[str, str]]:
    """CDT 2024 codes for a plan option (AAE CDT guide). Unknown tooth type keeps the placeholder digit."""
    out = []
    for item in _CDT.get(treatment, []):
        code = item["code"]
        if "{t}" in code:
            code = code.replace("{t}", _TOOTH_DIGIT.get(tooth_type, "x"))
        if "{e}" in code:
            code = code.replace("{e}", _TOOTH_DIGIT.get(tooth_type, "x"))
        out.append({"code": code, "descriptor": item["descriptor"],
                    "source": "AAE CDT 2024 guide",
                    "url": "https://www.aae.org/specialty/wp-content/uploads/sites/2/2023/12/CDTGuide_2024.pdf"})
    return out


class KnowledgeBase:
    def __init__(self, path: str = _PATH):
        data = json.load(open(path))
        self.version = data["version"]
        self.statements: List[Dict] = data["statements"]

    def for_tags(self, tags: Iterable[str]) -> List[Dict]:
        tags = set(tags)
        return [s for s in self.statements if tags & set(s["tags"])]

    def by_id(self, sid: str) -> Optional[Dict]:
        return next((s for s in self.statements if s["id"] == sid), None)


__all__ = ["KnowledgeBase", "cdt_codes_for", "tooth_type_from_iso"]
