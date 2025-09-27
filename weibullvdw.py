# weibullvdw.py - FIXED VERSION
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import gamma
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================
# PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND
# =============================================
st.set_page_config(
    page_title="Weibull Analysis Pro", 
    layout="wide",
    page_icon="🔬"
)

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
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
    }
    .plot-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# WEIBULL ANALYZER CLASS
# =============================================
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
            
            # Remove any infinite values
            valid_indices = np.isfinite(x) & np.isfinite(y)
            x_valid = x[valid_indices]
            y_valid = y[valid_indices]
            
            if len(x_valid) >= 2:
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                self.beta = slope
                self.eta = np.exp(-intercept / slope)
            else:
                # Fallback method
                self.beta = 1.5
                self.eta = np.mean(failures)
        else:
            # Single data point estimation
            self.beta = 1.5
            self.eta = failures[0] * 1.2
        
        # Ensure reasonable values
        self.beta = max(0.1, min(10.0, self.beta))
        self.eta = max(0.1, self.eta)
            
        return True
    
    def reliability_function(self, t):
        return np.exp(-(t/self.eta)**self.beta)
    
    def failure_rate(self, t):
        return (self.beta/self.eta) * (t/self.eta)**(self.beta-1)

# =============================================
# MAIN APPLICATION
# =============================================

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
        height=100,
        key="failure_data"
    )
    
    if st.button("📥 Load Sample Data", key="sample_btn"):
        st.session_state.failure_data = "120, 250, 300, 400, 560, 720, 900, 1250"
        st.rerun()

with col2:
    st.markdown("### ⚙️ CENSORED DATA")
    censored_data = st.text_area(
        "Enter censored times (still working):",
        placeholder="2000, 2500, 3000",
        height=100,
        key="censored_data"
    )

# Use sample data if loaded
if hasattr(st.session_state, 'failure_data'):
    failure_data = st.session_state.failure_data

if st.button("🚀 ANALYZE DATA", type="primary", use_container_width=True, key="analyze_btn"):
    if not failure_data:
        st.error("❌ Please enter failure data")
    else:
        try:
            # Parse data
            failures = []
            for x in failure_data.split(','):
                x_clean = x.strip()
                if x_clean:
                    try:
                        failures.append(float(x_clean))
                    except ValueError:
                        st.warning(f"Skipping invalid number: {x_clean}")
            
            censored = []
            if censored_data:
                for x in censored_data.split(','):
                    x_clean = x.strip()
                    if x_clean:
                        try:
                            censored.append(float(x_clean))
                        except ValueError:
                            st.warning(f"Skipping invalid number: {x_clean}")
            
            if not failures:
                st.error("❌ No valid failure data found")
            else:
                # Analyze
                analyzer = WeibullAnalyzer()
                analyzer.calculate_weibull(failures)
                
                st.success("✅ ANALYSIS COMPLETE!")
                
                # Calculate metrics
                mttf = analyzer.eta * gamma(1 + 1/analyzer.beta)
                b10_life = analyzer.eta * (-np.log(0.9)) ** (1/analyzer.beta)
                b50_life = analyzer.eta * (-np.log(0.5)) ** (1/analyzer.beta)
                
                # Display Metrics
                st.markdown("### 📈 RELIABILITY METRICS")
                
                # Create metric columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Shape Parameter</div>
                        <div class='metric-value'>β = {analyzer.beta:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Scale Parameter</div>
                        <div class='metric-value'>η = {analyzer.eta:.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>MTTF</div>
                        <div class='metric-value'>{mttf:.0f}</div>
                        <div class='metric-label'>{time_units}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>B10 Life</div>
                        <div class='metric-value'>{b10_life:.0f}</div>
                        <div class='metric-label'>{time_units}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Failure Phase Analysis
                st.markdown("### 🔍 FAILURE PATTERN ANALYSIS")
                if analyzer.beta < 1.0:
                    st.info("""
                    **📉 INFANT MORTALITY PHASE** (β < 1.0)
                    - **Pattern**: Decreasing failure rate over time
                    - **Meaning**: Early failures decreasing as weak units fail
                    - **Action**: Focus on burn-in testing and quality control
                    """)
                elif analyzer.beta <= 2.0:
                    st.success("""
                    **📊 USEFUL LIFE PHASE** (β ≈ 1.0-2.0)  
                    - **Pattern**: Relatively constant failure rate  
                    - **Meaning**: Random failures during normal operation  
                    - **Action**: Implement preventive maintenance schedule  
                    """)
                else:
                    st.warning("""
                    **📈 WEAR-OUT PHASE** (β > 2.0)
                    - **Pattern**: Increasing failure rate over time  
                    - **Meaning**: Aging and wear-out mechanisms dominant  
                    - **Action**: Plan for replacement and intensified inspections  
                    """)
                
                # Data Summary
                st.markdown("### 📋 DATA SUMMARY")
                st.code(f"""
                Failure Data: {len(failures)} points
                {', '.join(f'{x:.1f}' for x in sorted(failures))}
                
                Censored Data: {len(censored)} points  
                {', '.join(f'{x:.1f}' for x in sorted(censored)) if censored else 'None'}
                """)
                
                # Plots Section
                st.markdown("### 📊 RELIABILITY ANALYSIS PLOTS")
                
                # Reliability Plot
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                t_max = max(analyzer.eta * 3, max(failures) * 1.5) if failures else analyzer.eta * 3
                t = np.linspace(0.1, t_max, 200)
                reliability = analyzer.reliability_function(t)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=t, y=reliability, 
                    mode='lines', 
                    name='Reliability',
                    line=dict(width=4, color='#667eea')
                ))
                
                # Add B10 life marker
                fig1.add_vline(
                    x=b10_life, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="B10 Life", 
                    annotation_position="top right"
                )
                
                fig1.update_layout(
                    title='📈 Reliability Over Time',
                    xaxis_title=f'Time ({time_units})',
                    yaxis_title='Reliability Probability',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("""
                **📈 Understanding This Plot:**
                - The blue curve shows how reliability decreases from 100% to 0% over time
                - **B10 Life (red line)**: Time when 10% of units are expected to fail
                - **Steep drop** = Rapid reliability decrease | **Gentle slope** = Slow degradation
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Failure Rate Plot
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                failure_rate = analyzer.failure_rate(t)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=t, y=failure_rate, 
                    mode='lines', 
                    name='Failure Rate',
                    line=dict(width=4, color='#e74c3c')
                ))
                fig2.update_layout(
                    title='📊 Failure Rate Over Time',
                    xaxis_title=f'Time ({time_units})',
                    yaxis_title='Failure Rate',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown(f"""
                **📊 Failure Rate Pattern Analysis:**
                - **β = {analyzer.beta:.2f}** indicates {'decreasing' if analyzer.beta < 1.0 else 'constant' if analyzer.beta <= 2.0 else 'increasing'} failure rate
                - This helps identify your equipment's current life phase
                - Proper maintenance strategy depends on this pattern
                """)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate Report
                st.markdown("### 📄 PROFESSIONAL REPORT")
                report = f"""
# 🔬 WEIBULL ANALYSIS REPORT
## {company_name.upper()}

**Report Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Analyst:** {analyst_name}  
**Equipment:** {equipment_model}  
**Time Units:** {time_units}  

## EXECUTIVE SUMMARY

### Weibull Parameters:
- **Shape Parameter (β):** {analyzer.beta:.3f}
- **Scale Parameter (η):** {analyzer.eta:.1f} {time_units}

### Key Reliability Metrics:
- **Mean Time To Failure (MTTF):** {mttf:.1f} {time_units}
- **B10 Life (10% Failures):** {b10_life:.1f} {time_units}
- **B50 Life (Median Life):** {b50_life:.1f} {time_units}

## FAILURE PHASE ANALYSIS

### Identified Phase: {'Infant Mortality' if analyzer.beta < 1.0 else 'Useful Life' if analyzer.beta <= 2.0 else 'Wear-Out'}

**Characteristics:** {'Decreasing failure rate - early life failures' if analyzer.beta < 1.0 else 'Constant failure rate - normal operation' if analyzer.beta <= 2.0 else 'Increasing failure rate - aging equipment'}

## MAINTENANCE RECOMMENDATIONS

1. **Optimal Replacement Time:** {b10_life:.0f} {time_units}
2. **Preventive Maintenance Interval:** {b10_life/2:.0f} {time_units}
3. **Spare Parts Planning:** Maintain {max(2, len(failures))} units in inventory

## DATA SUMMARY

**Failure Data ({len(failures)} points):**
{', '.join(f'{x:.1f}' for x in sorted(failures))}

**Censored Data ({len(censored)} points):**
{', '.join(f'{x:.1f}' for x in sorted(censored)) if censored else 'None'}

---
*Generated by Professional Weibull Analysis System*
"""
                
                st.download_button(
                    label="📥 DOWNLOAD PROFESSIONAL REPORT",
                    data=report,
                    file_name=f"weibull_analysis_{equipment_model.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Analysis Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ by Vaibhav Engineering Solutions • Professional Reliability Engineering*")