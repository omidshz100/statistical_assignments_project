"""
Statistical Learning and Data Analysis - Assignment 2
Exponential vs Poisson Distribution Comparison

A comprehensive interactive app demonstrating the relationship between
Exponential and Poisson distributions through theory, simulation, and applications.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Exponential & Poisson Distribution Comparison",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 Exponential & Poisson Distribution Comparison</div>', 
            unsafe_allow_html=True)
st.markdown("### Statistical Learning and Data Analysis – Assignment 2")
st.markdown("---")

# ============================================================================
# SECTION 1: THEORETICAL COMPARISON
# ============================================================================

st.markdown('<div class="section-header">1️⃣ Theoretical Comparison</div>', 
            unsafe_allow_html=True)

# Create two columns for formulas
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔵 Poisson Distribution (Discrete)")
    st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots")
    st.markdown("""
    **Parameter:** λ > 0 (rate parameter)
    - Represents the **average number of events** per unit time
    - Used for counting events in fixed intervals
    """)
    st.latex(r"\text{Mean: } \mu = \lambda")
    st.latex(r"\text{Variance: } \sigma^2 = \lambda")

with col2:
    st.markdown("#### 🟢 Exponential Distribution (Continuous)")
    st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
    st.markdown("""
    **Parameter:** λ > 0 (rate parameter)
    - Represents the **rate of event occurrence**
    - Mean waiting time = 1/λ
    - Used for modeling time between events
    """)
    st.latex(r"\text{Mean: } \mu = \frac{1}{\lambda}")
    st.latex(r"\text{Variance: } \sigma^2 = \frac{1}{\lambda^2}")

# Relationship explanation
st.markdown("---")
st.markdown("#### 🔗 The Fundamental Relationship")
st.info("""
**Key Insight:** If events occur according to a **Poisson process** with rate λ, then:
- The **number of events** in time interval *t* follows **Poisson(λt)**
- The **waiting time between consecutive events** follows **Exponential(λ)**

These two distributions are intimately connected—one describes "how many," the other describes "how long until the next."
""")

# Convergence to Normal
st.markdown("#### 📈 Convergence to Normal Distribution")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Poisson → Normal**")
    st.write("As λ → ∞, Poisson(λ) ≈ Normal(λ, λ) by the Central Limit Theorem")
with col2:
    st.markdown("**Exponential Behavior**")
    st.write("Exponential remains right-skewed, but the sum of n i.i.d. Exponential(λ) ~ Gamma(n, λ) → Normal for large n")

# Visual comparison with multiple lambda values
st.markdown("---")
st.markdown("#### 📊 Visual Comparison for Different λ Values")

lambda_values = [0.5, 1, 2, 4]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, lam in enumerate(lambda_values):
    # Poisson plot (top row)
    ax_poisson = axes[0, idx]
    k_values = np.arange(0, int(lam * 4 + 15))
    pmf_values = stats.poisson.pmf(k_values, lam)
    ax_poisson.stem(k_values, pmf_values, basefmt=' ', linefmt='b-', markerfmt='bo')
    ax_poisson.set_title(f'Poisson(λ={lam})', fontsize=12, fontweight='bold')
    ax_poisson.set_xlabel('k (number of events)')
    ax_poisson.set_ylabel('P(X = k)')
    ax_poisson.grid(True, alpha=0.3)
    ax_poisson.axvline(lam, color='red', linestyle='--', linewidth=2, label=f'Mean = {lam}')
    ax_poisson.legend()
    
    # Exponential plot (bottom row)
    ax_exp = axes[1, idx]
    x_values = np.linspace(0, 6/lam, 500)
    pdf_values = stats.expon.pdf(x_values, scale=1/lam)
    ax_exp.plot(x_values, pdf_values, 'g-', linewidth=2)
    ax_exp.fill_between(x_values, pdf_values, alpha=0.3, color='green')
    ax_exp.set_title(f'Exponential(λ={lam})', fontsize=12, fontweight='bold')
    ax_exp.set_xlabel('x (waiting time)')
    ax_exp.set_ylabel('f(x)')
    ax_exp.grid(True, alpha=0.3)
    ax_exp.axvline(1/lam, color='red', linestyle='--', linewidth=2, label=f'Mean = {1/lam:.2f}')
    ax_exp.legend()

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""
**Observations:**
- **Poisson** becomes more symmetric and bell-shaped as λ increases
- **Exponential** always remains right-skewed, but the peak shifts left as λ increases
- Both distributions spread out as λ changes, reflecting their variance formulas
""")

# ============================================================================
# SECTION 2: SIMULATION STUDY
# ============================================================================

st.markdown('<div class="section-header">2️⃣ Simulation Study</div>', 
            unsafe_allow_html=True)

st.markdown("""
In this section, we generate 1000 random samples from both distributions and compare
the **empirical results** with **theoretical predictions**.
""")

# Lambda slider
lambda_sim = st.slider(
    "Choose λ (rate parameter):",
    min_value=0.1,
    max_value=5.0,
    value=2.0,
    step=0.1,
    help="Adjust the rate parameter to see how it affects both distributions"
)

# Caching simulation function
@st.cache_data
def generate_samples(lam, n_samples=1000, seed=42):
    """Generate samples from Poisson and Exponential distributions"""
    np.random.seed(seed)
    poisson_samples = stats.poisson.rvs(lam, size=n_samples)
    exponential_samples = stats.expon.rvs(scale=1/lam, size=n_samples)
    return poisson_samples, exponential_samples

# Generate samples
poisson_samples, exponential_samples = generate_samples(lambda_sim)

# Calculate statistics
poisson_mean_theory = lambda_sim
poisson_var_theory = lambda_sim
poisson_mean_sample = np.mean(poisson_samples)
poisson_var_sample = np.var(poisson_samples, ddof=1)

exp_mean_theory = 1 / lambda_sim
exp_var_theory = 1 / (lambda_sim ** 2)
exp_mean_sample = np.mean(exponential_samples)
exp_var_sample = np.var(exponential_samples, ddof=1)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Poisson histogram with theoretical PMF
ax1.hist(poisson_samples, bins=range(int(poisson_samples.max()) + 2), 
         density=True, alpha=0.6, color='blue', edgecolor='black', label='Sample Histogram')
k_range = np.arange(0, int(poisson_samples.max()) + 1)
pmf_theory = stats.poisson.pmf(k_range, lambda_sim)
ax1.plot(k_range, pmf_theory, 'ro-', markersize=8, linewidth=2, label='Theoretical PMF')
ax1.set_title(f'Poisson(λ={lambda_sim}) - 1000 Samples', fontsize=14, fontweight='bold')
ax1.set_xlabel('Value')
ax1.set_ylabel('Probability / Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Exponential histogram with theoretical PDF
ax2.hist(exponential_samples, bins=40, density=True, alpha=0.6, 
         color='green', edgecolor='black', label='Sample Histogram')
x_range = np.linspace(0, exponential_samples.max(), 500)
pdf_theory = stats.expon.pdf(x_range, scale=1/lambda_sim)
ax2.plot(x_range, pdf_theory, 'r-', linewidth=2, label='Theoretical PDF')
ax2.set_title(f'Exponential(λ={lambda_sim}) - 1000 Samples', fontsize=14, fontweight='bold')
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Comparison table
st.markdown("#### 📋 Statistical Comparison: Theory vs. Sample")
comparison_df = pd.DataFrame({
    'Distribution': ['Poisson', 'Exponential'],
    'Theoretical Mean': [f'{poisson_mean_theory:.4f}', f'{exp_mean_theory:.4f}'],
    'Sample Mean': [f'{poisson_mean_sample:.4f}', f'{exp_mean_sample:.4f}'],
    'Theoretical Variance': [f'{poisson_var_theory:.4f}', f'{exp_var_theory:.4f}'],
    'Sample Variance': [f'{poisson_var_sample:.4f}', f'{exp_var_sample:.4f}']
})

st.dataframe(comparison_df, use_container_width=True)

st.success("""
**Interpretation:** The sample moments (mean and variance) closely match the theoretical values,
confirming the correctness of our simulation. Small differences are due to random sampling variation,
which decreases as sample size increases (Law of Large Numbers).
""")

# ============================================================================
# SECTION 3: APPLICATION SCENARIO
# ============================================================================

st.markdown('<div class="section-header">3️⃣ Real-World Application Scenario</div>', 
            unsafe_allow_html=True)

st.markdown("""
### 🏦 Modeling Customer Arrivals at a Bank Help Desk

Consider a bank help desk where customers arrive randomly throughout the day.
We can model this system using both distributions:

- **Poisson Distribution:** Number of customers arriving in **one hour**
- **Exponential Distribution:** Time (in hours) **between consecutive arrivals**

**Assumption:** λ = 3 customers per hour (on average)
""")

# Simulation parameters
lambda_bank = 3
n_hours = 100
n_arrivals = 100

@st.cache_data
def simulate_bank_scenario(lam, n_hours, n_arrivals, seed=42):
    """Simulate bank customer arrivals"""
    np.random.seed(seed)
    hourly_counts = stats.poisson.rvs(lam, size=n_hours)
    inter_arrival_times = stats.expon.rvs(scale=1/lam, size=n_arrivals)
    return hourly_counts, inter_arrival_times

hourly_counts, inter_arrival_times = simulate_bank_scenario(lambda_bank, n_hours, n_arrivals)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Poisson: Hourly customer counts
ax1.hist(hourly_counts, bins=range(0, hourly_counts.max() + 2), 
         alpha=0.7, color='steelblue', edgecolor='black', density=False)
ax1.axvline(hourly_counts.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Sample Mean = {hourly_counts.mean():.2f}')
ax1.axvline(lambda_bank, color='orange', linestyle='--', 
            linewidth=2, label=f'Theoretical Mean = {lambda_bank}')
ax1.set_title('Number of Customers per Hour (100 hours)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Number of Customers')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Exponential: Inter-arrival times
ax2.hist(inter_arrival_times, bins=30, alpha=0.7, color='seagreen', 
         edgecolor='black', density=True)
x_range = np.linspace(0, inter_arrival_times.max(), 500)
pdf_theory = stats.expon.pdf(x_range, scale=1/lambda_bank)
ax2.plot(x_range, pdf_theory, 'r-', linewidth=2, label='Theoretical PDF')
ax2.axvline(inter_arrival_times.mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'Sample Mean = {inter_arrival_times.mean():.2f} hrs')
ax2.axvline(1/lambda_bank, color='orange', linestyle='--', 
            linewidth=2, label=f'Theoretical Mean = {1/lambda_bank:.2f} hrs')
ax2.set_title('Time Between Consecutive Arrivals (100 events)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Insights
st.markdown("#### 💡 Key Insights from the Bank Scenario")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Average Customers/Hour",
        value=f"{hourly_counts.mean():.2f}",
        delta=f"{hourly_counts.mean() - lambda_bank:.2f} vs theory"
    )

with col2:
    st.metric(
        label="Average Wait Time (hours)",
        value=f"{inter_arrival_times.mean():.3f}",
        delta=f"{inter_arrival_times.mean() - 1/lambda_bank:.3f} vs theory"
    )

with col3:
    st.metric(
        label="Max Customers in 1 Hour",
        value=f"{hourly_counts.max()}",
        delta=f"{hourly_counts.max() - lambda_bank} above average"
    )

st.markdown("---")

st.info("""
**Practical Implications:**
1. **Staffing Decisions:** Knowing the average arrival rate (λ=3/hour) helps determine how many staff members are needed
2. **Service Planning:** The exponential distribution tells us that ~63% of inter-arrival times are less than the mean (1/3 hour)
3. **Peak Detection:** Hours with significantly more than 3 customers may indicate special patterns or events
4. **Waiting Time:** If service time is also exponential, we can use queuing theory (M/M/1 model) for deeper analysis
""")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Statistical Learning and Data Analysis – Assignment 2</strong></p>
    <p>Exponential & Poisson Distribution Comparison</p>
    <p>Built with Streamlit, NumPy, SciPy, and Matplotlib</p>
</div>
""", unsafe_allow_html=True)

st.markdown("#### 🎛️ Try your own λ (shape explorer)")

lambda_shape = st.slider(
    "Adjust λ to see how both shapes move:",
    min_value=0.1,
    max_value=5.0,
    value=1.5,
    step=0.1,
    key="lambda_shape"
)

k_max = max(12, int(lambda_shape * 4 + 15))
k_vals = np.arange(0, k_max + 1)
pmf_shape = stats.poisson.pmf(k_vals, lambda_shape)

x_vals = np.linspace(0, 6 / lambda_shape, 500)
pdf_shape = stats.expon.pdf(x_vals, scale=1 / lambda_shape)

fig_shape, (axp, axe) = plt.subplots(1, 2, figsize=(12, 4))

axp.stem(k_vals, pmf_shape, basefmt=" ", linefmt="b-", markerfmt="bo")
axp.axvline(lambda_shape, color="red", linestyle="--", linewidth=2, label=f"Mean = {lambda_shape:.2f}")
axp.set_title(f"Poisson(λ={lambda_shape:.2f})", fontsize=12, fontweight="bold")
axp.set_xlabel("k (number of events)")
axp.set_ylabel("P(X = k)")
axp.grid(True, alpha=0.3)
axp.legend()

axe.plot(x_vals, pdf_shape, "g-", linewidth=2, label="PDF")
axe.fill_between(x_vals, pdf_shape, alpha=0.3, color="green")
axe.axvline(1 / lambda_shape, color="red", linestyle="--", linewidth=2, label=f"Mean = {1/lambda_shape:.2f}")
axe.set_title(f"Exponential(λ={lambda_shape:.2f})", fontsize=12, fontweight="bold")
axe.set_xlabel("x (waiting time)")
axe.set_ylabel("f(x)")
axe.grid(True, alpha=0.3)
axe.legend()

plt.tight_layout()
st.pyplot(fig_shape)
plt.close()

st.caption("Move λ to see how the Poisson mass shifts and the Exponential peak slides left/right.")