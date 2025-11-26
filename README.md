

https://github.com/user-attachments/assets/62349c5d-37ed-4d0c-b067-69a32290db5f

 # **AI-Driven Financial Stress Test Agent**

The **AI-Driven Financial Stress Test Agent** is a full-stack application built to automate and accelerate financial risk analysis. It leverages a **Small Language Model (SLM)** to translate narrative stress-test scenarios (e.g., *“What happens if a global recession hits?”*) into **quantifiable macroeconomic shocks**, which are then processed by a high-performance **FastAPI ML engine**.

The engine simulates a dynamic risk model (mocked **XGBoost + Monte Carlo PnL simulation**) to compute financial metrics such as:

* **Expected Loss (EL)**
* **Value at Risk (VaR)**
* **Stressed total net impact**
* **Simulated PnL trajectories**

The system is designed as a lightweight, modular microservice architecture — combining conversational AI, structured scenario generation, and explainable risk modeling.

---

## 🚀 **Key Features**

### **🔹 SLM-Powered Scenario Generation**

Converts natural language inputs into validated, structured **Pydantic scenario models**, including shocks such as:

* GDP contraction
* Equity market drawdown
* Interest-rate shifts
* Scenario time horizons

### **🔹 FastAPI-Based ML Engine**

Handles intensive risk computations:

* Mocked XGBoost-style model
* Monte-Carlo PnL simulation
* Stressed EL & VaR calculations

### **🔹 Streamlit Frontend**

A clean, conversational UI that provides:

* Real-time scenario explanations
* Health checks for the backend API
* Interactive charts (PnL paths, impact bars) using Plotly

### **🔹 Decoupled Architecture**

Frontend and backend run independently, enabling:

* Scalability
* Faster iteration
* Deployment flexibility

---

## 🛠️ **Architecture Overview**

| Component           | Technology                  | Purpose                                                             |
| ------------------- | --------------------------- | ------------------------------------------------------------------- |
| **SLM Client App**  | Streamlit, Plotly, Requests | Conversational interface, scenario visualization, API communication |
| **ML Engine API**   | FastAPI, Uvicorn, Pydantic  | Scenario validation, risk calculations, response formatting         |
| **Risk Model Core** | Python, Numpy, Pandas       | Mock XGBoost logic, Monte Carlo PnL engine                          |

---

## 📦 **Project Structure**

```
ai_stress_test_agent/
├── ml_engine_api/
│   ├── ml_engine.py                # Risk model & simulation logic
│   ├── risk_api_models.py          # Pydantic request/response schemas
│   └── ml_engine_api.py            # FastAPI routes & server setup
├── hf_client_app/
│   ├── app.py                      # Streamlit UI & visualization
│   └── hf_agent_client.py          # SLM agent + API connector
├── .env                            # Environment variables
└── requirements.txt                # Python dependencies
```

---

## ⚙️ **Installation & Setup**

### **Prerequisites**

* Python **3.10+**
* `pip`

### **1. Clone the Repository**

```bash
git clone https://github.com/narc-kany/AI-Driven-Financial-Stress-Test-Agent.git
cd AI-Driven-Financial-Stress-Test-Agent
```

### **2. Create and Activate a Virtual Environment**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Configure Environment Variables**

In `.env`:

```
ML_ENGINE_API_URL="http://127.0.0.1:8000"
```

---

## ▶️ **Running the Application**

> **Important:**
> Run both the FastAPI backend and the Streamlit frontend from the **directory ABOVE** the `ai_stress_test_agent/` package.

---

### **1. Start the ML Engine API (Backend)**

```bash
python -m uvicorn ai_stress_test_agent.ml_engine_api.ml_engine_api:app \
  --reload --host 127.0.0.1 --port 8000
```

You should see:

```
INFO:     Application startup complete.
```

---

### **2. Start the SLM Client App (Frontend)**

Open a **new terminal**, activate the environment, and run:

```bash
python -m streamlit run ai_stress_test_agent/hf_client_app/app.py
```

Streamlit should launch at:

**[http://localhost:8501](http://localhost:8501)**

---

## 💬 **Example Usage**

Try a scenario like:

> *“Simulate a severe global recession where GDP falls by 4% for 12 months and equity markets drop 35%. Estimate the Expected Loss and overall portfolio impact.”*

The system will:

1. **Generate structured macroeconomic shocks** via the SLM
2. **Send the scenario to the ML Engine API**
3. Display:

   * Stressed Expected Loss
   * Total net impact
   * PnL simulation curve
   * Feature/Factor contribution breakdown

---

## 🧭 **Roadmap (Planned Enhancements)**

* Plug-in architecture for real XGBoost, CatBoost, and LightGBM models
* Multi-scenario batch processing
* Support for CCAR-style regulatory scenarios
* More extensive explainability (SHAP-like outputs)
* Auth & user management

---
