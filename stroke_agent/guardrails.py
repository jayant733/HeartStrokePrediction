"""
Guardrails for AI Agent to ensure safe medical advice.
"""
def validate_output(agent_response):
    if "diagnose" in agent_response.lower() or "prescribe" in agent_response.lower():
        return "WARNING: AI cannot officially diagnose or prescribe. Please consult a human doctor."
    return agent_response
