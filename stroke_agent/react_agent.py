"""
Agentic AI using ReAct (Reason and Act) prompting.
"""
def run_react_agent(patient_data):
    # 1. Reason: "I need to check the patient's blood pressure."
    # 2. Act: Call Tool -> get_vitals()
    # 3. Reason: "Blood pressure is high. I should predict stroke risk."
    # 4. Act: Call Tool -> predict_stroke()
    print("Executing ReAct workflow...")
    return "High Risk"
