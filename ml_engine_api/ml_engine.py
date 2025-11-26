import time
import numpy as np
import pandas as pd
from typing import List, Dict
from ai_stress_test_agent.ml_engine_api.risk_api_models import ScenarioInput, RiskMetric, RiskOutput, MacroFactorShock

class FinancialModelEngine:
    """
    A service class simulating an advanced, dynamic financial risk model (e.g., XGBoost on default rates, 
    coupled with Monte Carlo simulation for PnL).
    """
    def __init__(self):
        # In a real environment, this would load a trained XGBoost model and pre-processor
        # self.model = xgb.Booster()
        # self.model.load_model("trained_financial_model.json")
        self.is_loaded = True
        print("INFO: Financial Model Engine initialized (using mock model).")

    def _simulate_pnl(self, shocks: List[MacroFactorShock], horizon: int) -> List[float]:
        """
        Mocks a dynamic PnL (Profit and Loss) simulation based on macro factor shocks.
        The severity of the shock correlates with the magnitude of the loss path.
        """
        # Calculate a cumulative stress factor from the input shocks
        stress_factor = sum(s.shock_value for s in shocks if 'Rate' in s.factor_name or 'Index' in s.factor_name) * 0.1
        
        # Base PnL path (starting at 0)
        base_pnl = np.zeros(horizon)
        
        # Simulate market noise and trend
        np.random.seed(42)
        random_walk = np.cumsum(np.random.normal(0, 0.5, horizon)) - np.linspace(0, 0.5 * horizon, horizon)
        
        # Apply stress: A deep, continuous drop, scaled by the stress factor
        stress_path = -np.cumsum(np.linspace(0.1, 0.5 + abs(stress_factor), horizon))
        
        simulated_pnl = base_pnl + random_walk + stress_path
        
        return [float(x) for x in simulated_pnl]

    def _mock_xgboost_prediction(self, scenario: ScenarioInput, pnl_path: List[float]) -> Dict[str, float]:
        """
        Mocks the core risk calculation logic that would normally be handled by an ML model.
        It uses the shock magnitude and PnL simulation to derive key metrics.
        """
        total_shock_magnitude = sum(abs(s.shock_value) for s in scenario.shocks)
        final_pnl = pnl_path[-1] if pnl_path else 0
        
        # --- Mocked XGBoost/ML Logic ---
        
        # 1. Expected Loss (EL): High shock -> High EL
        expected_loss = (50 + total_shock_magnitude * 5) * 1.5 + abs(final_pnl) * 0.1 
        
        # 2. Value at Risk (VaR 99%): Derived from the simulated PnL distribution
        sim_data = np.array(pnl_path)
        # VaR is typically the 1% quantile of the loss distribution (or 99% of the PnL distribution)
        # We'll use the minimum simulated value as a proxy for the worst-case VaR
        var_99 = abs(min(sim_data) * 1.2) * 2 

        # 3. Liquidity Gap: Scales with time horizon and scenario severity
        liquidity_gap = total_shock_magnitude * 3 + scenario.time_horizon_months * 0.5
        
        # --- Mocked Feature Impact (Explainability) ---
        feature_impact = {
            s.factor_name: round(s.shock_value * 1.5, 2)
            for s in scenario.shocks
        }
        feature_impact["Model_Non_Linearity"] = round(np.mean(np.diff(pnl_path)) * 10, 2)
        
        return {
            "expected_loss": expected_loss,
            "var_99": var_99,
            "liquidity_gap": liquidity_gap,
            "feature_impact": feature_impact,
            "net_impact": expected_loss + var_99
        }

    def run_stress_test(self, scenario: ScenarioInput) -> RiskOutput:
        """
        Orchestrates the dynamic stress test: simulation + ML prediction.
        """
        if not self.is_loaded:
            raise Exception("Model not loaded.")

        # 1. Dynamic Simulation (e.g., Monte Carlo)
        pnl_path = self._simulate_pnl(scenario.shocks, scenario.time_horizon_months)
        
        # 2. ML Prediction and Risk Metric Calculation
        mock_results = self._mock_xgboost_prediction(scenario, pnl_path)
        
        metrics = [
            RiskMetric(
                metric_name="Expected_Loss (1yr)",
                value=mock_results["expected_loss"],
                unit="USD Million",
                baseline_value=15.0  # Base EL is 15M
            ),
            RiskMetric(
                metric_name="VaR 99%",
                value=mock_results["var_99"],
                unit="USD Million",
                baseline_value=30.0  # Base VaR is 30M
            ),
            RiskMetric(
                metric_name="Liquidity Gap",
                value=mock_results["liquidity_gap"],
                unit="Days",
                baseline_value=60.0 # Base Liquidity gap is 60 days
            )
        ]

        # Generate a unique ID for the run
        scenario_id = f"SCN-{int(time.time())}"
        
        # Construct the final output
        output = RiskOutput(
            scenario_id=scenario_id,
            status="Success",
            description=f"Stress test completed for {scenario.portfolio_segment} under the '{scenario.scenario_name}' scenario.",
            net_impact=mock_results["net_impact"],
            metrics=metrics,
            pnl_simulation_path=pnl_path,
            feature_impact=mock_results["feature_impact"]
        )
        
        return output

# Initialize the model globally (as a singleton instance in a real app)
engine = FinancialModelEngine()