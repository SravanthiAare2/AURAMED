🏥  AURAMED: An Explainable AI system for Personalized Drug & Dosage Recommendation

AURAMED is an Explainable AI-based clinical decision support system designed to recommend personalized drugs and dosages using patient health data. The system analyzes clinical parameters and predicts the most suitable treatment while assessing Adverse Drug Reaction (ADR) risk.

Problem Statement

Selecting the correct drug and dosage depends on multiple clinical factors. Manual evaluation can be complex and time-consuming, and many AI systems lack transparency. There is a need for an intelligent and explainable system to support safer medical decisions.

Proposed Solution

AURAMED uses machine learning models to:

Recommend the appropriate drug

Predict dosage levels

Estimate ADR risk

Provide alternate medication options The system also uses SHAP-based Explainable AI to show the factors influencing predictions and generates a structured clinical report.

System Architecture

Input Features: Age, Gender, Weight, BP, Blood Glucose, Cholesterol, CRP, eGFR, ALT, Severity, Comorbidities, Symptoms

Models Used:

Random Forest Classifier – Drug & ADR Prediction

Random Forest Regressor – Dosage Prediction

SHAP – Model Explainability

Output:

Recommended Drug

Dosage (mg)

ADR Risk Level

Visual Explanations

Clinical Report (PDF)

Future Scope

The system can be improved by integrating real hospital datasets, electronic health records (EHR), advanced AI models, and expanding support for multiple diseases to make it more scalable and intelligent.

Conclusion

AURAMED enhances treatment accuracy, improves patient safety, and demonstrates the potential of explainable AI in modern healthcare decision-making.
