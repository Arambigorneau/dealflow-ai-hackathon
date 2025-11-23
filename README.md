# ⚖️ DealFlow AI - M&A Due Diligence Agent

![IBM watsonx](https://img.shields.io/badge/Powered%20by-IBM%20watsonx-blue)
![Status](https://img.shields.io/badge/Status-Prototype-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

**DealFlow AI** is an autonomous agent designed to revolutionize the Mergers & Acquisitions (M&A) Due Diligence process. By leveraging **IBM watsonx Orchestrate** and the latest **IBM Granite 3.3** model, it automates the analysis of complex legal contracts, identifying critical risks (penalties, termination clauses) in seconds instead of hours.

Made by Ménahëze Ewan on 11/22/2025 (ewanmenaheze.fr/menewa2904@gmail.com)

Submitted for the Agentic AI Hackathon with IBM watsonx Orchestrate

---

## 🏗️ Architecture & IBM Integration

This solution follows a **Hybrid Agentic Architecture**:

```mermaid
graph TD
    A[User / Lawyer] -->|Uploads PDF| B(Streamlit Interface)
    A -->|Chat Command| C(IBM watsonx Orchestrate Agent)
    
    B -->|API Call| D{IBM watsonx.ai}
    C -->|Invokes Skill| D
    
    D -->|Inference| E[Model: IBM Granite 3.3 Instruct]
    E -->|JSON Output| B
    E -->|Natural Language| C

