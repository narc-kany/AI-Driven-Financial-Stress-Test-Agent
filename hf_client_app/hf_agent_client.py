import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv
import os
import time
from ai_stress_test_agent.ml_engine_api.risk_api_models import ScenarioInput, RiskOutput, MacroFactorShock

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
ML_ENGINE_API_URL = os.getenv("ML_ENGINE_API_URL", "http://127.0.0.1:8000")

# --- MOCK LLM FUNCTIONALITY ---

def mock_llm_scenario_generation(user_prompt: str) -> Dict[str, Any]:
    """
    Mocks an SLM (Small Language Model) API call to generate a structured 
    financial stress test scenario (JSON output compliant with ScenarioInput Pydantic model).
    
    In a real app, this would use the Gemini/Hugging Face API with Function Calling 
    or JSON mode to ensure structured output.
    """
    # Simulate thinking time
    time.sleep(1.5) 
    
    # A sophisticated prompt analysis would happen here. For mocking, we check keywords.
    prompt_lower = user_prompt.lower()
    
    if "recession" in prompt_lower or "severe" in prompt_lower:
        # Severe Recession Scenario
        shocks = [
            {"factor_name": "GDP_Growth", "shock_value": -3.5, "unit": "pct_points"},
            {"factor_name": "Unemployment_Rate", "shock_value": 4.0, "unit": "pct_points"},
            {"factor_name": "Equity_Index_Return", "shock_value": -30.0, "unit": "relative_change"},
        ]
        scenario_name = "Severe Global Recession"
        narrative = "A sharp, synchronized global economic downturn triggered by high inflation and aggressive central bank tightening, leading to a surge in default rates and market volatility. This scenario tests credit and market risk simultaneously."
    elif "interest rate" in prompt_lower or "inflation" in prompt_lower:
        # Stagflation/Interest Rate Shock Scenario
        shocks = [
            {"factor_name": "Interest_Rate_3M", "shock_value": 3.0, "unit": "pct_points"},
            {"factor_name": "Inflation_CPI", "shock_value": 5.0, "unit": "pct_points"},
            {"factor_name": "Housing_Price_Index", "shock_value": -10.0, "unit": "relative_change"},
        ]
        scenario_name = "Stagflation Crisis"
        narrative = "Prolonged high inflation coupled with stagnant growth forces central banks to hike rates significantly. This severely pressures fixed-income portfolios and increases debt servicing costs for businesses and consumers."
    else:
        # Default/Milder Scenario
        shocks = [
            {"factor_name": "Commodity_Prices", "shock_value": -15.0, "unit": "relative_change"},
            {"factor_name": "Exchange_Rate_Vol", "shock_value": 0.5, "unit": "pct_points"},
        ]
        scenario_name = "Mild Market Correction"
        narrative = "A generalized but mild market correction driven by supply chain normalization and geopolitical calm, primarily affecting commodity-linked portfolios. This is a baseline test for operational and commodity risk."

    # Construct the raw data dictionary
    raw_scenario_data = {
        "scenario_name": scenario_name,
        "narrative": narrative,
        "shocks": shocks,
        "portfolio_segment": "Mixed_Portfolio",
        "time_horizon_months": 24 if "long" in prompt_lower else 12,
    }
    
    return raw_scenario_data

# --- FASTAPI CONNECTOR FUNCTIONALITY ---

def run_ml_stress_test(scenario_data: Dict[str, Any]) -> RiskOutput:
    """
    Sends the LLM-generated scenario to the FastAPI ML Engine for computation.
    """
    endpoint = f"{ML_ENGINE_API_URL}/run_stress_test"
    
    try:
        # Validate data against the Pydantic model before sending (optional but good practice)
        ScenarioInput.model_validate(scenario_data)
        
        # Send POST request to the ML Engine
        response = requests.post(endpoint, json=scenario_data, timeout=30)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Validate and return the response using Pydantic
        risk_output_data = response.json()
        return RiskOutput.model_validate(risk_output_data)

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to ML Engine API at {endpoint}: {e}")
        return RiskOutput(
            scenario_id="ERROR-API",
            status="Failure",
            description=f"Could not connect to the ML Engine API. Is the backend running? Error: {e}",
            net_impact=0.0,
            metrics=[]
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return RiskOutput(
            scenario_id="ERROR-UNKNOWN",
            status="Failure",
            description=f"An unexpected error occurred during processing. Error: {e}",
            net_impact=0.0,
            metrics=[]
        )