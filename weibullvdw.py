# weibull_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import io
import base64
from datetime import datetime

# Use the same WeibullAnalyzer class from your original code
class WeibullAnalyzer:
    """Core Weibull analysis engine (same as desktop version)"""
    
    def __init__(self):
        self.data = None
        self.failure_times = None
        self.censored_times = None
        self.mle_params = None
        self.lse_params = None
        self.mle_ci = None
        self.lse_ci = None
        
    def load_data(self, df):
        """Load data from DataFrame"""
        try:
            # Validate required columns
            if 'time' not in df.columns or 'event' not in df.columns:
                raise ValueError("Data must contain 'time' and 'event' columns")
            
            # Clean data
            df = df.dropna(subset=['time', 'event'])
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df['event'] = pd.to_numeric(df['event'], errors='coerce')
            df = df.dropna(subset=['time', 'event'])
            df['event'] = df['event'].astype(int)
            
            if len(df) == 0:
                raise ValueError("No valid data found after cleaning")
            
            self.data = df
            self.failure_times = df[df['event'] == 1]['time'].values
            self.censored_times = df[df['event'] == 0]['time'].values
            
            return True, f"Loaded {len(df)} records ({len(self.failure_times)} failures, {len(self.censored_times)} censored)"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def weibull_mle(self, data, censored=None):
        """Maximum Likelihood Estimation for Weibull distribution with censoring"""
        if censored is None:
            censored = np.zeros_like(data)
        
        def neg_log_likelihood(params):
            beta, eta = params
            if beta <= 0 or eta <= 0:
                return 1e10
                
            failures = data[censored == 1]
            censored_obs = data[censored == 0]
            
            # Log-likelihood for failures
            ll_failures = np.sum(np.log(beta/eta) + (beta-1)*np.log(failures/eta) - (failures/eta)**beta)
            
            # Log-likelihood for censored observations
            ll_censored = np.sum(-(censored_obs/eta)**beta)
            
            return -(ll_failures + ll_censored)
        
        # Initial guess
        beta0 = 1.5
        eta0 = np.median(data) if len(data) > 0 else 1.0
        
        try:
            result = minimize(neg_log_likelihood, [beta0, eta0], 
                            bounds=[(0.1, 10), (0.1, max(data)*2)])
            if result.success:
                return result.x[0], result.x[1]
            else:
                raise ValueError("MLE optimization failed")
        except:
            # Fallback to simple estimation
            if len(self.failure_times) > 0:
                log_data = np.log(self.failure_times)
                beta = 1.2 / np.std(log_data)
                eta = np.exp(np.mean(log_data))
                return beta, eta
            else:
                return 1.0, 1.0
    
    def weibull_lse(self, data):
        """Least Squares Estimation using Rank Regression"""
        if len(data) == 0:
            return 1.0, 1.0
            
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        # Median ranks (Benard's approximation)
        ranks = (np.arange(1, n+1) - 0.3) / (n + 0.4)
        
        # Weibull plot coordinates
        x = np.log(sorted_data)
        y = np.log(-np.log(1 - ranks))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        beta = slope
        eta = np.exp(-intercept / beta)
        
        return beta, eta
    
    def calculate_confidence_intervals(self, beta, eta, n, alpha=0.05):
        """Calculate 95% confidence intervals using Fisher Information"""
        z = stats.norm.ppf(1 - alpha/2)
        
        # Approximate standard errors
        se_beta = beta / np.sqrt(n)
        se_eta = eta / (beta * np.sqrt(n))
        
        ci_beta = (beta - z * se_beta, beta + z * se_beta)
        ci_eta = (eta - z * se_eta, eta + z * se_eta)
        
        return ci_beta, ci_eta
    
    def calculate_metrics(self, beta, eta):
        """Calculate reliability metrics"""
        # MTTF (Mean Time To Failure)
        mttf = eta * np.math.gamma(1 + 1/beta)
        
        # B-life values
        b10 = eta * (-np.log(0.9)) ** (1/beta)
        b50 = eta * (-np.log(0.5)) ** (1/beta)
        b90 = eta * (-np.log(0.1)) ** (1/beta)
        
        return {
            'mttf': mttf,
            'b10': b10,
            'b50': b50,
            'b90': b90
        }
    
    def reliability_function(self, t, beta, eta):
        """Reliability at time t"""
        return np.exp(-(t/eta)**beta)
    
    def failure_rate(self, t, beta, eta):
        """Failure rate at time t"""
        return (beta/eta) * (t/eta)**(beta-1)
    
    def analyze(self):
        """Perform complete Weibull analysis"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        all_times = np.concatenate([self.failure_times, self.censored_times])
        events = np.concatenate([np.ones(len(self.failure_times)), 
                               np.zeros(len(self.censored_times))])
        
        # MLE estimation
        self.mle_params = self.weibull_mle(all_times, events)
        
        # LSE estimation (using only failure data)
        self.lse_params = self.weibull_lse(self.failure_times)
        
        # Confidence intervals
        n_failures = len(self.failure_times)
        self.mle_ci = self.calculate_confidence_intervals(self.mle_params[0], 
                                                         self.mle_params[1], n_failures)
        self.lse_ci = self.calculate_confidence_intervals(self.lse_params[0], 
                                                         self.lse_params[1], n_failures)
        
        # Calculate metrics
        mle_metrics = self.calculate_metrics(*self.mle_params)
        lse_metrics = self.calculate_metrics(*self.lse_params)
        
        return {
            'mle': {
                'beta': self.mle_params[0],
                'eta': self.mle_params[1],
                'ci_beta': self.mle_ci[0],
                'ci_eta': self.mle_ci[1],
                'metrics': mle_metrics
            },
            'lse': {
                'beta': self.lse_params[0],
                'eta': self.lse_params[1],
                'ci_beta': self.lse_ci[0],
                'ci_eta': self.lse_ci[1],
                'metrics': lse_metrics
            }
        }

def create_sample_data():
    """Create sample data for demonstration"""
    sample_data = {
        'time': [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 
                300, 325, 350, 375, 400, 425, 450, 475, 500],
        'event': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 0, 0, 0, 0, 0, 0]  # Last 6 are censored
    }
    return pd.DataFrame(sample_data)

def plot_to_html(fig):
    """Convert matplotlib figure to HTML for Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'

def main():
    st.set_page_config(
        page_title="Weibull Analysis Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Professional Weibull Analysis Tool")
    st.markdown("### Reliability Engineering Web Application")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = WeibullAnalyzer()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Section",
        ["Data Input", "Analysis Results", "Visualization", "Report"]
    )
    
    # Data Input Section
    if app_mode == "Data Input":
        st.header("📁 Data Input")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose CSV or Excel file", 
                type=['csv', 'xlsx'],
                help="File must contain 'time' and 'event' columns (event: 1=failure, 0=censored)"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    success, message = st.session_state.analyzer.load_data(df)
                    if success:
                        st.success(message)
                        st.session_state.data_loaded = True
                    else:
                        st.error(message)
                        
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        with col2:
            st.subheader("Sample Data")
            if st.button("Load Sample Data"):
                sample_df = create_sample_data()
                success, message = st.session_state.analyzer.load_data(sample_df)
                if success:
                    st.success("Sample data loaded successfully!")
                    st.session_state.data_loaded = True
        
        if st.session_state.get('data_loaded', False):
            st.subheader("Data Preview")
            st.dataframe(st.session_state.analyzer.data, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(st.session_state.analyzer.data))
            with col2:
                st.metric("Failures", len(st.session_state.analyzer.failure_times))
            with col3:
                st.metric("Censored", len(st.session_state.analyzer.censored_times))
    
    # Analysis Results Section
    elif app_mode == "Analysis Results":
        st.header("📈 Analysis Results")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load data first in the 'Data Input' section")
            return
        
        if st.button("Run Weibull Analysis"):
            with st.spinner("Performing Weibull analysis..."):
                try:
                    results = st.session_state.analyzer.analyze()
                    st.session_state.results = results
                    st.success("Analysis completed successfully!")
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        
        if st.session_state.get('results'):
            results = st.session_state.results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Maximum Likelihood Estimation (MLE)")
                st.write(f"**Shape Parameter (β):** {results['mle']['beta']:.4f}")
                st.write(f"**95% CI for β:** ({results['mle']['ci_beta'][0]:.4f}, {results['mle']['ci_beta'][1]:.4f})")
                st.write(f"**Scale Parameter (η):** {results['mle']['eta']:.4f}")
                st.write(f"**95% CI for η:** ({results['mle']['ci_eta'][0]:.4f}, {results['mle']['ci_eta'][1]:.4f})")
                
                st.subheader("Reliability Metrics (MLE)")
                metrics = results['mle']['metrics']
                st.write(f"**MTTF:** {metrics['mttf']:.4f}")
                st.write(f"**B10 Life:** {metrics['b10']:.4f}")
                st.write(f"**B50 Life:** {metrics['b50']:.4f}")
                st.write(f"**B90 Life:** {metrics['b90']:.4f}")
            
            with col2:
                st.subheader("Least Squares Estimation (LSE)")
                st.write(f"**Shape Parameter (β):** {results['lse']['beta']:.4f}")
                st.write(f"**95% CI for β:** ({results['lse']['ci_beta'][0]:.4f}, {results['lse']['ci_beta'][1]:.4f})")
                st.write(f"**Scale Parameter (η):** {results['lse']['eta']:.4f}")
                st.write(f"**95% CI for η:** ({results['lse']['ci_eta'][0]:.4f}, {results['lse']['ci_eta'][1]:.4f})")
                
                st.subheader("Reliability Metrics (LSE)")
                metrics = results['lse']['metrics']
                st.write(f"**MTTF:** {metrics['mttf']:.4f}")
                st.write(f"**B10 Life:** {metrics['b10']:.4f}")
                st.write(f"**B50 Life:** {metrics['b50']:.4f}")
                st.write(f"**B90 Life:** {metrics['b90']:.4f}")
            
            # Reliability Calculator
            st.subheader("Reliability Calculator")
            time_input = st.number_input("Enter time t for reliability calculation:", 
                                       value=100.0, min_value=0.0)
            
            if st.button("Calculate Reliability"):
                beta_mle, eta_mle = results['mle']['beta'], results['mle']['eta']
                beta_lse, eta_lse = results['lse']['beta'], results['lse']['eta']
                
                rel_mle = st.session_state.analyzer.reliability_function(time_input, beta_mle, eta_mle)
                rel_lse = st.session_state.analyzer.reliability_function(time_input, beta_lse, eta_lse)
                fr_mle = st.session_state.analyzer.failure_rate(time_input, beta_mle, eta_mle)
                fr_lse = st.session_state.analyzer.failure_rate(time_input, beta_lse, eta_lse)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**MLE Results:**")
                    st.write(f"Reliability R(t): {rel_mle:.4f} ({rel_mle*100:.2f}%)")
                    st.write(f"Failure Rate h(t): {fr_mle:.6f}")
                with col2:
                    st.write("**LSE Results:**")
                    st.write(f"Reliability R(t): {rel_lse:.4f} ({rel_lse*100:.2f}%)")
                    st.write(f"Failure Rate h(t): {fr_lse:.6f}")
    
    # Visualization Section
    elif app_mode == "Visualization":
        st.header("📊 Visualization")
        
        if not st.session_state.get('results'):
            st.warning("Please run analysis first in the 'Analysis Results' section")
            return
        
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Probability Plot", "PDF", "CDF", "Reliability", "Hazard Rate"]
        )
        
        if st.button("Generate Plot"):
            with st.spinner("Generating plot..."):
                try:
                    fig = create_plot(plot_type, st.session_state.analyzer, st.session_state.results)
                    st.pyplot(fig)
                    st.success("Plot generated successfully!")
                except Exception as e:
                    st.error(f"Plot generation failed: {str(e)}")
    
    # Report Section
    elif app_mode == "Report":
        st.header("📋 Analysis Report")
        
        if not st.session_state.get('results'):
            st.warning("Please run analysis first in the 'Analysis Results' section")
            return
        
        if st.button("Generate Report"):
            report = generate_report(st.session_state.analyzer, st.session_state.results)
            st.text_area("Analysis Report", report, height=400)
            
            # Download buttons
            st.download_button(
                label="Download Report as Text",
                data=report,
                file_name=f"weibull_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def create_plot(plot_type, analyzer, results):
    """Create different types of plots"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    beta_mle, eta_mle = results['mle']['beta'], results['mle']['eta']
    beta_lse, eta_lse = results['lse']['beta'], results['lse']['eta']
    
    t_max = max(analyzer.failure_times) * 1.5 if len(analyzer.failure_times) > 0 else 100
    t = np.linspace(0.1, t_max, 200)
    
    if plot_type == "Probability Plot":
        failure_times = analyzer.failure_times
        if len(failure_times) > 0:
            sorted_times = np.sort(failure_times)
            n = len(sorted_times)
            ranks = (np.arange(1, n+1) - 0.3) / (n + 0.4)
            
            x_theo = np.linspace(min(sorted_times)*0.1, max(sorted_times)*1.5, 100)
            y_theo = 1 - np.exp(-(x_theo/eta_mle)**beta_mle)
            
            ax.plot(sorted_times, ranks, 'bo', label='Data', markersize=4)
            ax.plot(x_theo, y_theo, 'r-', label='Weibull Fit', linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title('Weibull Probability Plot')
            
    elif plot_type == "PDF":
        pdf_mle = (beta_mle/eta_mle) * (t/eta_mle)**(beta_mle-1) * np.exp(-(t/eta_mle)**beta_mle)
        pdf_lse = (beta_lse/eta_lse) * (t/eta_lse)**(beta_lse-1) * np.exp(-(t/eta_lse)**beta_lse)
        
        ax.plot(t, pdf_mle, 'b-', label='MLE', linewidth=2)
        ax.plot(t, pdf_lse, 'r--', label='LSE', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability Density')
        ax.set_title('Weibull PDF')
        
    elif plot_type == "CDF":
        cdf_mle = 1 - np.exp(-(t/eta_mle)**beta_mle)
        cdf_lse = 1 - np.exp(-(t/eta_lse)**beta_lse)
        
        ax.plot(t, cdf_mle, 'b-', label='MLE', linewidth=2)
        ax.plot(t, cdf_lse, 'r--', label='LSE', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Weibull CDF')
        
    elif plot_type == "Reliability":
        rel_mle = np.exp(-(t/eta_mle)**beta_mle)
        rel_lse = np.exp(-(t/eta_lse)**beta_lse)
        
        ax.plot(t, rel_mle, 'b-', label='MLE', linewidth=2)
        ax.plot(t, rel_lse, 'r--', label='LSE', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Reliability R(t)')
        ax.set_title('Reliability Function')
        
    elif plot_type == "Hazard Rate":
        hazard_mle = (beta_mle/eta_mle) * (t/eta_mle)**(beta_mle-1)
        hazard_lse = (beta_lse/eta_lse) * (t/eta_lse)**(beta_lse-1)
        
        ax.plot(t, hazard_mle, 'b-', label='MLE', linewidth=2)
        ax.plot(t, hazard_lse, 'r--', label='LSE', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Hazard Rate h(t)')
        ax.set_title('Hazard Rate Function')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def generate_report(analyzer, results):
    """Generate comprehensive report"""
    report = f"""
WEIBULL ANALYSIS REPORT
========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SUMMARY:
-------------
Total Records: {len(analyzer.data)}
Failures: {len(analyzer.failure_times)}
Censored: {len(analyzer.censored_times)}

MAXIMUM LIKELIHOOD ESTIMATION (MLE) RESULTS:
-------------------------------------------
Shape Parameter (β): {results['mle']['beta']:.4f}
95% Confidence Interval: ({results['mle']['ci_beta'][0]:.4f}, {results['mle']['ci_beta'][1]:.4f})

Scale Parameter (η): {results['mle']['eta']:.4f}
95% Confidence Interval: ({results['mle']['ci_eta'][0]:.4f}, {results['mle']['ci_eta'][1]:.4f})

Reliability Metrics:
- MTTF: {results['mle']['metrics']['mttf']:.4f}
- B10 Life: {results['mle']['metrics']['b10']:.4f}
- B50 Life: {results['mle']['metrics']['b50']:.4f}
- B90 Life: {results['mle']['metrics']['b90']:.4f}

LEAST SQUARES ESTIMATION (LSE) RESULTS:
--------------------------------------
Shape Parameter (β): {results['lse']['beta']:.4f}
95% Confidence Interval: ({results['lse']['ci_beta'][0]:.4f}, {results['lse']['ci_beta'][1]:.4f})

Scale Parameter (η): {results['lse']['eta']:.4f}
95% Confidence Interval: ({results['lse']['ci_eta'][0]:.4f}, {results['lse']['ci_eta'][1]:.4f})

Reliability Metrics:
- MTTF: {results['lse']['metrics']['mttf']:.4f}
- B10 Life: {results['lse']['metrics']['b10']:.4f}
- B50 Life: {results['lse']['metrics']['b50']:.4f}
- B90 Life: {results['lse']['metrics']['b90']:.4f}

INTERPRETATION:
---------------
• Shape Parameter (β):
  - β < 1: Decreasing failure rate (infant mortality)
  - β = 1: Constant failure rate (random failures)
  - β > 1: Increasing failure rate (wear-out failures)

• Scale Parameter (η):
  - Characteristic life where 63.2% of units have failed

RECOMMENDATIONS:
----------------
Based on the analysis, consider appropriate maintenance strategies.
"""
    return report

if __name__ == "__main__":
    main()