"""Multi-agent case conference for ECR diagnosis and treatment-plan suggestion (research)."""
from .blackboard import Blackboard, Evidence
from .criteria import CRITERIA, HYPOTHESES, PRIOR
from .llm_backend import LocalHFBackend, LLMBackend, NullBackend, OpenAICompatibleBackend, PerplexitySDKBackend, auto_backend
from .orchestrator import CaseConference, consensus_markdown, record_diagnosis_feedback
from .roles import (AnatomyAgent, ClassifierAgent, ClinicalCorrelationAgent, ImagingQAAgent, LesionScreeningAgent,
                    RadiologistAgent, SkepticAgent, TreatmentPlannerAgent)
from .clinical import ClinicalFindings
from .criteria import DIAGNOSTIC_QUESTIONS, answer_questions

__all__ = ["Blackboard", "Evidence", "CRITERIA", "HYPOTHESES", "PRIOR", "LLMBackend", "NullBackend",
           "OpenAICompatibleBackend", "PerplexitySDKBackend", "auto_backend", "CaseConference",
           "consensus_markdown", "record_diagnosis_feedback", "AnatomyAgent", "ClassifierAgent", "ImagingQAAgent",
           "LesionScreeningAgent", "RadiologistAgent", "SkepticAgent", "TreatmentPlannerAgent",
           "ClinicalCorrelationAgent", "ClinicalFindings", "DIAGNOSTIC_QUESTIONS", "answer_questions"]
