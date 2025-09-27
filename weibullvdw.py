# weibullvdw.py - PROFESSIONAL WEIBULL ANALYSIS TOOL
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import weibull_min
from scipy.special import gamma
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================
# PROFESSIONAL CSS DESIGN
# =============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    .plot-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class WeibullAnalyzer:
    def __init__(self):
        self.beta = None
        self.eta = None
        
    def calculate_weibull(self, failures):
        if len(failures) == 0:
            raise ValueError("Please enter failure data")
        
        # Professional Weibull estimation
        if len(failures) > 1:
            sorted_data = np.sort(failures)
            n = len(sorted_data)
            ranks = (np.arange(1, n+1) - 0.3) / (n + 0.4)
            x = np.log(sorted_data)
            y = np.log(-np.log(1 - ranks))
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            self.beta = slope
            self.eta = np.exp(-intercept / slope)
        else:
            # Single data point estimation
            self.beta = 1.5
            self.eta = failures[0] * 1.2
            
        return True
    
    def reliability_function(self, t):
        return np.exp(-(t/self.eta)**self.beta)
    
    def failure_rate(self, t):
        return (self.beta/self.eta) * (t/self.eta)**(self.beta-1)

def main():
    st.set_page_config(page_title="Weibull Analysis Pro", layout="wide")
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🔬 PROFESSIONAL WEIBULL ANALYSIS</h1>
        <h3>Reliability Engineering Made Simple • Vaibhav Engineering Solutions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏢 COMPANY INFO")
        company_name = st.text_input("Company Name", value="Vaibhav Engineering Solutions")
        analyst_name = st.text_input("Analyst Name", value="Vaibhav Wagh")
        equipment_model = st.text_input("Equipment Model", value="Motor Drive X500")
        time_units = st.selectbox("Time Units", ["hours", "days", "weeks", "months", "years"])
    
    # Main Content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 FAILURE DATA")
        failure_data = st.text_area(
            "Enter failure times (comma separated):",
            placeholder="120, 250, 300, 400, 560, 720, 900",
            height=100
        )
        
        if st.button("📥 Load Sample Data"):
            failure_data = "120, 250, 300, 400, 560, 720, 900, 1250"
    
    with col2:
        st.markdown("### ⚙️ CENSORED DATA")
        censored_data = st.text_area(
            "Enter censored times (still working):",
            placeholder="2000, 2500, 3000",
            height=100
        )
    
    if st.button("🚀 ANALYZE DATA", type="primary", use_container_width=True):
        if not failure_data:
            st.error("❌ Please enter failure data")
        else:
            try:
                # Parse data
                failures = [float(x.strip()) for x in failure_data.split(',') if x.strip()]
                censored = [float(x.strip()) for x in censored_data.split(',')] if censored_data else []
                
                # Analyze
                analyzer = WeibullAnalyzer()
                analyzer.calculate_weibull(failures)
                
                st.success("✅ ANALYSIS COMPLETE!")
                
                # Display Metrics
                st.markdown("### 📈 RESULTS")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div>Shape Parameter</div>
                        <div class='metric-value'>β = {analyzer.beta:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div>Scale Parameter</div>
                        <div class='metric-value'>η = {analyzer.eta:.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    mttf = analyzer.eta * gamma(1 + 1/analyzer.beta)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div>MTTF</div>
                        <div class='metric-value'>{mttf:.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    b10_life = analyzer.eta * (-np.log(0.9)) ** (1/analyzer.beta)
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div>B10 Life</div>
                        <div class='metric-value'>{b10_life:.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Failure Phase Analysis
                st.markdown("### 🔍 FAILURE PATTERN")
                if analyzer.beta < 1.0:
                    st.info("**📉 INFANT MORTALITY**: Decreasing failure rate - early life failures")
                elif analyzer.beta <= 2.0:
                    st.success("**📊 USEFUL LIFE**: Constant failure rate - normal operation")
                else:
                    st.warning("**📈 WEAR-OUT**: Increasing failure rate - aging equipment")
                
                # Plots
                st.markdown("### 📊 RELIABILITY ANALYSIS")
                
                # Reliability Plot
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                t = np.linspace(0.1, analyzer.eta * 3, 200)
                reliability = analyzer.reliability_function(t)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=t, y=reliability, mode='lines', name='Reliability',
                                        line=dict(width=3, color='#667eea')))
                fig1.update_layout(title='Reliability Over Time', 
                                 xaxis_title=f'Time ({time_units})', 
                                 yaxis_title='Reliability')
                st.plotly_chart(fig1, use_container_width=True)
                st.markdown("""
                **📈 Understanding Reliability**: This curve shows how reliability decreases from 100% to 0% over time. 
                The steeper the drop, the faster your equipment loses reliability.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Failure Rate Plot
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                failure_rate = analyzer.failure_rate(t)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=t, y=failure_rate, mode='lines', name='Failure Rate',
                                        line=dict(width=3, color='#e74c3c')))
                fig2.update_layout(title='Failure Rate Over Time', 
                                 xaxis_title=f'Time ({time_units})', 
                                 yaxis_title='Failure Rate')
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown(f"""
                **📊 Failure Rate Pattern**: 
                - β = {analyzer.beta:.2f} indicates {'decreasing' if analyzer.beta < 1.0 else 'constant' if analyzer.beta <= 2.0 else 'increasing'} failure rate
                - This helps identify the equipment's life phase
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate Report
                report = f"""
# WEIBULL ANALYSIS REPORT
**Company:** {company_name}
**Analyst:** {analyst_name}
**Equipment:** {equipment_model}
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## RESULTS
- Shape Parameter (β): {analyzer.beta:.2f}
- Scale Parameter (η): {analyzer.eta:.0f} {time_units}
- MTTF: {mttf:.0f} {time_units}
- B10 Life: {b10_life:.0f} {time_units}

## RECOMMENDATIONS
Based on β = {analyzer.beta:.2f}, this equipment is in the {'infant mortality' if analyzer.beta < 1.0 else 'useful life' if analyzer.beta <= 2.0 else 'wear-out'} phase.

Maintenance should focus on {'burn-in testing and early monitoring' if analyzer.beta < 1.0 else 'preventive maintenance' if analyzer.beta <= 2.0 else 'replacement planning'}.
"""
                
                st.download_button(
                    label="📥 DOWNLOAD REPORT",
                    data=report,
                    file_name=f"weibull_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()