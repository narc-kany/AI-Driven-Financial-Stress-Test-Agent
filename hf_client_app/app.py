import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ai_stress_test_agent.hf_client_app.hf_agent_client import mock_llm_scenario_generation, run_ml_stress_test
from ai_stress_test_agent.ml_engine_api.risk_api_models import ScenarioInput, RiskOutput

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Financial Stress Test Agent",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "# An advanced SLM-driven stress test application."}
)

# --- HELPER FUNCTIONS ---

def display_risk_results(risk_output: RiskOutput):
    """Displays the stress test results using cards, charts, and tables."""
    
    # 1. High-Level Summary Cards
    st.subheader("📊 Stress Test Summary")
    
    col1, col2, col3 = st.columns(3)
    
    # Total Impact Card
    with col1:
        st.metric(
            label="Total Net Impact (Loss)", 
            value=f"${risk_output.net_impact:,.2f}M", 
            delta_color="inverse", 
            delta="Severe" if risk_output.net_impact > 100 else "Moderate"
        )
        st.caption(f"Scenario ID: {risk_output.scenario_id}")

    # Key Metric Cards
    metrics_df = pd.DataFrame([m.model_dump() for m in risk_output.metrics])
    
    if "Expected_Loss (1yr)" in metrics_df['metric_name'].values:
        el_metric = metrics_df[metrics_df['metric_name'] == "Expected_Loss (1yr)"].iloc[0]
        change = el_metric['value'] - el_metric['baseline_value']
        with col2:
            st.metric(
                label="Stressed Expected Loss (EL)",
                value=f"${el_metric['value']:,.2f}M",
                delta=f"▲ {change:,.2f}M vs Baseline",
                delta_color="inverse"
            )

    if "VaR 99%" in metrics_df['metric_name'].values:
        var_metric = metrics_df[metrics_df['metric_name'] == "VaR 99%"].iloc[0]
        change = var_metric['value'] - var_metric['baseline_value']
        with col3:
            st.metric(
                label="Stressed VaR 99%",
                value=f"${var_metric['value']:,.2f}M",
                delta=f"▲ {change:,.2f}M vs Baseline",
                delta_color="inverse"
            )

    # 2. Dynamic PnL Simulation Chart
    st.subheader("📉 Dynamic PnL Simulation Path")
    
    if risk_output.pnl_simulation_path:
        pnl_df = pd.DataFrame({
            "Month": range(1, len(risk_output.pnl_simulation_path) + 1),
            "Cumulative PnL (M)": risk_output.pnl_simulation_path
        })
        
        fig_pnl = px.line(
            pnl_df, 
            x="Month", 
            y="Cumulative PnL (M)", 
            title=f"PnL Trajectory over {len(risk_output.pnl_simulation_path)} Months",
            color_discrete_sequence=['#FF4B4B'] # Red/Danger color
        )
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Zero PnL")
        fig_pnl.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig_pnl, use_container_width=True)

    # 3. Model Explainability and Detailed Metrics
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🔑 ML Model Feature Impact")
        if risk_output.feature_impact:
            impact_df = pd.DataFrame(
                list(risk_output.feature_impact.items()), 
                columns=['Factor', 'Impact_Score']
            ).sort_values(by='Impact_Score', ascending=False)
            
            fig_impact = px.bar(
                impact_df, 
                x='Impact_Score', 
                y='Factor', 
                orientation='h', 
                title="Top Driving Factors (XGBoost Feature Importance)",
                color='Impact_Score',
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig_impact.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_impact, use_container_width=True)
        else:
            st.info("No feature impact data available for this run.")

    with col_b:
        st.subheader("📈 Detailed Risk Metrics")
        # Format the metrics for display
        metrics_df['Value'] = metrics_df.apply(lambda row: f"{row['value']:,.2f} {row['unit']}", axis=1)
        metrics_df['Baseline'] = metrics_df.apply(lambda row: f"{row['baseline_value']:,.2f} {row['unit']}", axis=1)
        metrics_df['Change'] = metrics_df['value'] - metrics_df['baseline_value']
        
        display_cols = ['metric_name', 'Value', 'Baseline', 'Change']
        st.dataframe(metrics_df[display_cols], use_container_width=True, hide_index=True)


def handle_user_prompt(prompt):
    """Processes the user prompt through the SLM and ML Engine."""
    
    # 1. SLM Agent: Generate Structured Scenario
    with st.status("🧠 **SLM Agent is generating scenario...**", expanded=True) as status:
        status.update(label=f"SLM Agent analyzing prompt: '{prompt[:50]}...'")
        
        # Call the mocked LLM function
        raw_scenario_data = mock_llm_scenario_generation(prompt)
        
        # Display generated scenario for transparency
        scenario_input = ScenarioInput.model_validate(raw_scenario_data)
        st.session_state.scenario_input = scenario_input
        
        st.markdown(f"**Scenario Name:** {scenario_input.scenario_name}")
        st.markdown(f"**Narrative:** {scenario_input.narrative}")
        st.dataframe(pd.DataFrame([s.model_dump() for s in scenario_input.shocks]), use_container_width=True, hide_index=True)
        
        status.update(label="Scenario generation complete. Submitting to ML Engine...", state="running")
        
        # 2. ML Engine: Run Stress Test
        risk_output = run_ml_stress_test(raw_scenario_data)
        st.session_state.risk_output = risk_output
        
        if risk_output.status == "Success":
            status.update(label="✅ Stress test completed successfully.", state="complete", expanded=False)
            st.success(f"Stress Test Run {risk_output.scenario_id} Complete!")
        else:
            status.update(label="❌ Stress test failed.", state="error")
            st.error(risk_output.description)


# --- MAIN STREAMLIT APP LOGIC ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Welcome! I am the Financial Stress Test Agent. Describe a scenario you want to simulate (e.g., 'What if a severe global recession hits the bond market?')."}
    )
if "scenario_input" not in st.session_state:
    st.session_state.scenario_input = None
if "risk_output" not in st.session_state:
    st.session_state.risk_output = None


st.title("AI-Driven Financial Stress Test Agent 🤖")
st.markdown("Use the chat below to define a complex scenario. The SLM Agent will structure the macroeconomic shocks and run them through the Dynamic Risk Model (XGBoost backend).")

# --- MAIN LAYOUT: CHATBOX AND RESULTS ---
col_chat, col_results = st.columns([1, 2], gap="large")

# --- CHAT INTERFACE ---
with col_chat:
    st.subheader("Agent Conversation")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Define a stress scenario..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the prompt (calls SLM and ML Engine)
        with st.chat_message("assistant"):
            handle_user_prompt(prompt)
            
            # Add final response to chat history
            if st.session_state.risk_output and st.session_state.risk_output.status == "Success":
                 final_response = (
                    f"I have successfully generated and run the scenario: **{st.session_state.scenario_input.scenario_name}**.\n\n"
                    f"The total calculated loss (Net Impact) is **${st.session_state.risk_output.net_impact:,.2f} Million**.\n\n"
                    "The detailed results and dynamic PnL path are displayed in the 'Stress Test Results' panel on the right. You can now ask for adjustments, like 'Make the interest rate shock 1% higher' or 'Run a milder correction test'."
                )
            else:
                final_response = f"The stress test failed. Reason: {st.session_state.risk_output.description}"
            
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})

# --- RESULTS DISPLAY ---
with col_results:
    st.subheader("Stress Test Results")
    
    if st.session_state.risk_output and st.session_state.risk_output.status == "Success":
        st.success(f"Scenario: {st.session_state.scenario_input.scenario_name}")
        display_risk_results(st.session_state.risk_output)
    elif st.session_state.risk_output and st.session_state.risk_output.status == "Failure":
         st.error(f"Test Failed. See Chat for details: {st.session_state.risk_output.description}")
    else:
        st.info("Awaiting scenario input. Start a conversation with the Agent on the left to run your first dynamic stress test.")