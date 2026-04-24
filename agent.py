import os
from typing import Dict, Literal

import instructor
from groq import Groq
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


def disable_broken_local_proxy():
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    for key in proxy_keys:
        value = os.environ.get(key)
        if value and ("127.0.0.1:9" in value or "localhost:9" in value):
            os.environ.pop(key, None)


def load_local_env():
    candidates = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(BASE_DIR, "env"),
        os.path.join(PROJECT_ROOT, ".env"),
        os.path.join(PROJECT_ROOT, "env"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ[key] = value


def get_groq_client():
    disable_broken_local_proxy()
    load_local_env()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")

    raw = Groq(api_key=api_key)
    wrapped = instructor.from_groq(raw)
    return raw, wrapped


raw_client, client = get_groq_client()
MODEL_CANDIDATES = [
    os.getenv("GROQ_MODEL"),
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
MODEL_CANDIDATES = [m for m in MODEL_CANDIDATES if m]


class Violation(BaseModel):
    file: str
    component: str
    issue: str
    rule_type: str
    constraint_value: str
    code: str


class RiskAssessment(BaseModel):
    severity: Literal["Critical", "High", "Medium", "Low"]
    business_impact: str
    recommended_action: Literal["Create_Jira_Ticket", "Draft_GitHub_PR"]


class ImpactExplanation(BaseModel):
    business_impact: str


def compute_severity(violation: Violation) -> str:
    rule_type = violation.rule_type.lower()
    if rule_type == "limit":
        return "Critical"
    if "hour" in violation.constraint_value.lower():
        return "High"
    if rule_type == "prohibition":
        return "Critical"
    return "Medium"


def decide_action(violation: Violation, severity: str) -> str:
    simple_patterns = ["=", "limit", "hours"]
    if any(p in violation.code.lower() for p in simple_patterns):
        return "Draft_GitHub_PR"
    return "Create_Jira_Ticket"


def generate_business_impact(violation: Violation, severity: str) -> str:
    prompt = f"""
Explain the business impact of this RBI compliance violation.

Rule Type: {violation.rule_type}
Issue: {violation.issue}
Severity: {severity}

Focus on:
- RBI penalties
- customer impact
- financial/legal consequences

Be concise (2-3 lines max).
"""

    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            response = client.chat.completions.create(
                model=model,
                response_model=ImpactExplanation,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.business_impact
        except Exception as e:
            last_error = e
            print(f"[WARN] Model {model} failed, trying next...")

    print("[ERROR] All Groq models failed:", last_error)
    return "Potential regulatory penalties and customer impact."


def assess_risk(violation_data: Dict) -> RiskAssessment:
    print("\n[PHASE 4] Risk Engine")

    violation = Violation(**violation_data)
    severity = compute_severity(violation)
    action = decide_action(violation, severity)
    impact = generate_business_impact(violation, severity)

    return RiskAssessment(
        severity=severity,
        business_impact=impact,
        recommended_action=action,
    )


def tool_create_jira_ticket(violation: Violation, assessment: RiskAssessment):
    print("\n[ACTION] Creating Jira Ticket...")

    payload = {
        "project": "COMPLIANCE",
        "summary": f"[AUTO] RBI Violation in {violation.file}",
        "description": f"""
Issue: {violation.issue}
Severity: {assessment.severity}
Impact: {assessment.business_impact}
Component: {violation.component}
""",
        "priority": assessment.severity,
    }

    print("JIRA PAYLOAD:")
    print(payload)
    return "JIRA-123"


def tool_draft_github_pr(violation: Violation, assessment: RiskAssessment):
    print("\n[ACTION] Drafting GitHub PR...")

    fixed_code = violation.code.replace("12", "24").replace("50000", "100000")

    payload = {
        "title": f"fix: compliance update in {violation.file}",
        "body": f"""
Auto-generated compliance fix.

Issue: {violation.issue}
Impact: {assessment.business_impact}

Suggested Fix:
{fixed_code}
""",
        "draft": True,
    }

    print("PR PAYLOAD:")
    print(payload)
    return "PR-456"


def orchestrate(violation_data: Dict):
    print("\n[START] Starting Compliance Agent")

    violation = Violation(**violation_data)
    assessment = assess_risk(violation_data)

    print("\n[RESULT]")
    print(f"Severity: {assessment.severity}")
    print(f"Impact: {assessment.business_impact}")
    print(f"Action: {assessment.recommended_action}")

    if assessment.recommended_action == "Draft_GitHub_PR":
        result = tool_draft_github_pr(violation, assessment)
        print("\n[HITL] Draft PR created. Awaiting approval.")
    else:
        result = tool_create_jira_ticket(violation, assessment)

    return result


if __name__ == "__main__":
    sample_violation = {
        "file": "notification_service.py",
        "component": "send_pre_transaction_notification",
        "issue": "Lead time 12h < required 24h",
        "rule_type": "obligation",
        "constraint_value": "24 hours",
        "code": "lead_time_hours = 12",
    }
    orchestrate(sample_violation)
