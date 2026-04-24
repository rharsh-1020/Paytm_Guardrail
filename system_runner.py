import argparse
import os
from pathlib import Path
from typing import Dict, List


def _read_code_snippet(project_root: Path, rel_file: str, max_lines: int = 80) -> str:
    target = project_root / rel_file
    if not target.exists():
        return ""
    try:
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[:max_lines])
    except Exception:
        return ""


def run_system(run_phase1: bool, run_phase2: bool, run_phase3: bool, run_phase4: bool):
    project_root = Path(__file__).resolve().parent.parent
    print("[SYSTEM] Starting unified compliance system")

    phase2_policy = None
    phase3_result: Dict = {"violations": []}
    phase4_results: List[Dict] = []

    if run_phase1:
        print("\n[SYSTEM] Phase 1: Scraping + Embedding")
        from main import run_pipeline

        run_pipeline()

    if run_phase2:
        print("\n[SYSTEM] Phase 2: Regulatory Reasoning")
        from reasoning import run_reasoning_pipeline

        phase2_policy = run_reasoning_pipeline()
        print(f"[SYSTEM] Extracted constraints: {len(phase2_policy.constraints)}")

    if run_phase3:
        print("\n[SYSTEM] Phase 3: Code Intelligence")
        from code_intelligence import run_code_intelligence_pipeline

        rules = None
        if phase2_policy and phase2_policy.constraints:
            # Feed Phase-2 constraints into Phase-3 checks.
            rules = [
                {
                    "rule_type": c.rule_type,
                    "constraint_value": c.constraint_value or "",
                }
                for c in phase2_policy.constraints
            ]

        phase3_result = run_code_intelligence_pipeline(rules=rules)
        print(f"[SYSTEM] Violations found: {len(phase3_result.get('violations', []))}")

    if run_phase4 and phase3_result.get("violations"):
        print("\n[SYSTEM] Phase 4: Jira/PR Agent Routing")
        try:
            from agent import orchestrate
        except Exception as e:
            print(f"[SYSTEM][WARN] Phase 4 skipped (agent init failed): {e}")
            orchestrate = None

        if orchestrate:
            for v in phase3_result["violations"]:
                payload = {
                    "file": v.get("file", ""),
                    "component": v.get("component", ""),
                    "issue": v.get("issue", ""),
                    "rule_type": v.get("rule_type", ""),
                    "constraint_value": str(v.get("constraint_value", "")),
                    "code": _read_code_snippet(project_root, v.get("file", "")),
                }
                result = orchestrate(payload)
                phase4_results.append({"violation": payload, "result": result})

    print("\n[SYSTEM] Completed")
    return {
        "phase2_constraints": len(phase2_policy.constraints) if phase2_policy else 0,
        "phase3_violations": len(phase3_result.get("violations", [])),
        "phase4_actions": len(phase4_results),
    }


def main():
    parser = argparse.ArgumentParser(description="Unified compliance system runner")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    parser.add_argument("--skip-phase4", action="store_true")
    args = parser.parse_args()

    summary = run_system(
        run_phase1=not args.skip_phase1,
        run_phase2=not args.skip_phase2,
        run_phase3=not args.skip_phase3,
        run_phase4=not args.skip_phase4,
    )
    print(f"[SYSTEM] Summary: {summary}")


if __name__ == "__main__":
    main()
