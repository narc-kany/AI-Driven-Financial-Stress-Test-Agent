from fastapi import FastAPI, HTTPException
from ai_stress_test_agent.ml_engine_api.risk_api_models import ScenarioInput, RiskOutput
from ai_stress_test_agent.ml_engine_api.ml_engine import engine # Import the initialized engine
from contextlib import asynccontextmanager
import uvicorn

# Use the lifespan manager for initialization (FastAPI >= 0.100.0)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure the engine is ready
    if not engine.is_loaded:
        print("WARNING: Model not fully initialized, running mock.")
    print("INFO: ML Engine API Server starting up.")
    yield
    # Shutdown: Cleanup if needed
    print("INFO: ML Engine API Server shutting down.")

app = FastAPI(
    title="Financial AI Stress Test Engine",
    description="Backend service for running dynamic financial stress tests based on SLM-generated scenarios and an XGBoost/Dynamic Risk Model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "engine_ready": engine.is_loaded}

@app.post("/run_stress_test", response_model=RiskOutput)
async def run_stress_test_endpoint(scenario: ScenarioInput):
    """
    Receives a structured scenario from the SLM agent and executes the stress test.
    """
    try:
        print(f"Received scenario: {scenario.scenario_name}")
        result = engine.run_stress_test(scenario)
        return result
    except Exception as e:
        print(f"Error during stress test: {e}")
        raise HTTPException(status_code=500, detail=f"Stress test calculation failed: {str(e)}")

# To run this file directly for development:
# if __name__ == "__main__":
#     # NOTE: The host/port should match the ML_ENGINE_API_URL in .env
#     uvicorn.run(app, host="127.0.0.1", port=8000)