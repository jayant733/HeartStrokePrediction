"""
Human-in-the-loop (HITL) workflow for critical predictions.
"""
def require_human_approval(prediction):
    if prediction == "High Risk":
        approval = input("Human Doctor, do you approve this prediction? (y/n): ")
        if approval.lower() != 'y':
            return "Prediction overridden by human."
    return prediction
