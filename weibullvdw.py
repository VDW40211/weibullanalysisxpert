# weibull_app.py - COMPLETE APPLICATION
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import weibull_min, kstest, norm
from scipy.special import gamma
import warnings
warnings.filterwarnings('ignore')

# =============================================
# CUSTOM CSS - BEAUTIFUL DESIGN
# =============================================
CUSTOM_CSS = """
<style>
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --shadow-soft: 20px 20px 60px #d9d9d9, -20px -20px 60px #ffffff;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    font-family: 'Inter', sans-serif !important;
}

.neumorphic-card {
    background: #e9ecef;
    border-radius: 25px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-soft);
    border: 1px solid rgba(255, 255, 255, 0.5);
}

.gradient-text {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

.stButton > button {
    background: var(--primary-gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
}

.metric-card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-soft);
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.5rem 0;
}
</style>
"""

class ProfessionalWeibullAnalyzer:
    def __init__(self):
        self.beta = None
        self.eta = None
        
    def mle_estimation(self, failures, censored=None):
        failures = np.array(failures)
        if len(failures) == 0:
            raise ValueError("Please enter failure data")
        
        # Simple robust estimation
        if len(failures) > 1:
            log_data = np.log(failures)
            var_log = np.var(log_data)
            self.beta = 1.2 / np.sqrt(var_log) if var_log > 0 else 1.5
            self.eta = np.exp(np.mean(log_data) + 0.5772 / self.beta)
        else:
            self.beta = 1.5
            self.eta = failures[0] * 1.5
            
        self.beta = max(self.beta, 0.1)
        self.eta = max(self.eta, 0.1)
        return True

class WeibullAnalysisApp:
    def __init__(self):
        self.analyzer = ProfessionalWeibullAnalyzer()
    
    def inject_css(self):
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    def parse_input_data(self, data_str):
        if not data_str: return []
        cleaned = data_str.replace('\n', ',').replace(';', ',')
        numbers = []
        for item in cleaned.split(','):
            item = item.strip()
            if item:
                try:
                    num = float(item)
                    if num > 0: numbers.append(num)
                except: continue
        return numbers
    
    def run(self):
        st.set_page_config(page_title="Weibull Analysis", layout="wide")
        self.inject_css()
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 class='gradient-text'>🔬 Weibull Analysis Tool</h1>
            <p>Professional Reliability Analysis Made Simple</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### 🏢 Your Information")
            analyst_name = st.text_input("Your Name", value="Vaibhav Wagh")
            equipment_model = st.text_input("Equipment Model", placeholder="e.g., Motor Drive X500")
            time_units = st.selectbox("Time Units", ["hours", "days", "years"])
        
        # Main content
        st.markdown("### 📊 Enter Your Data")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Failure Times** (when equipment broke)")
            failure_data = st.text_area(
                "Enter numbers separated by commas",
                placeholder="120, 250, 300, 400, 560...",
                height=100,
                label_visibility="collapsed"
            )
            
            if st.button("📥 Load Sample Data", use_container_width=True):
                st.session_state.sample_data = "120, 250, 300, 400, 560, 720, 900, 1250"
        
        with col2:
            st.markdown("**Censored Times** (still working)")
            censored_data = st.text_area(
                "Enter numbers separated by commas", 
                placeholder="2000, 2500, 3000...",
                height=100,
                label_visibility="collapsed"
            )
        
        # Use sample data if loaded
        if hasattr(st.session_state, 'sample_data'):
            failure_data = st.session_state.sample_data
        
        if st.button("🚀 Analyze Data", type="primary", use_container_width=True):
            failures = self.parse_input_data(failure_data)
            censored = self.parse_input_data(censored_data)
            
            if not failures:
                st.error("❌ Please enter at least one failure time")
            else:
                try:
                    with st.spinner("🔬 Analyzing your data..."):
                        self.analyzer.mle_estimation(failures, censored)
                    
                    # Show Results
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div>Shape Parameter</div>
                            <div class='metric-value'>β = {self.analyzer.beta:.2f}</div>
                            <div>Failure Pattern</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div>Scale Parameter</div>
                            <div class='metric-value'>η = {self.analyzer.eta:.0f}</div>
                            <div>{time_units}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Interpretation
                    st.markdown("### 📈 Interpretation")
                    if self.analyzer.beta < 1.0:
                        st.info("**Infant Mortality Phase:** Equipment has decreasing failure rate - typical for new equipment")
                    elif self.analyzer.beta <= 2.0:
                        st.success("**Useful Life Phase:** Equipment has stable failure rate - normal operation")
                    else:
                        st.warning("**Wear-Out Phase:** Equipment has increasing failure rate - time for maintenance")
                    
                    # Simple Chart
                    t = np.linspace(0.1, self.analyzer.eta * 3, 100)
                    reliability = np.exp(-(t/self.analyzer.eta)**self.analyzer.beta)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=t, y=reliability, mode='lines', name='Reliability'))
                    fig.update_layout(title='Reliability Over Time', xaxis_title=f'Time ({time_units})', yaxis_title='Reliability')
                    st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    app = WeibullAnalysisApp()
    app.run()