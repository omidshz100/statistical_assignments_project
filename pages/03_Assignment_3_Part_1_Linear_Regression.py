import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

st.title("Linear Regression Simulation and Diagnostics")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")
n = st.sidebar.number_input("Sample size (n)", value=300, min_value=50, max_value=1000)
p = st.sidebar.number_input("Number of predictors (p)", value=10, min_value=2, max_value=20)
noise_std = st.sidebar.slider("Noise std deviation", 0.1, 5.0, 1.0, 0.1)
seed = st.sidebar.number_input("Random seed", value=42, min_value=0)

# True coefficients
st.header("1. Data Generation")
st.write("**True Model:**")
st.latex(r"y = \beta_0 + \sum_{i=1}^{p} \beta_i x_i + \epsilon, \quad \epsilon \sim N(0, \sigma^2)")

true_beta = np.zeros(p)
true_beta[0] = 5.0   # Strong effect
true_beta[1] = 3.0   # Strong effect
true_beta[2] = -2.5  # Moderate effect
true_beta[3] = 1.5   # Moderate effect
true_beta[4] = 0.5   # Weak effect
# Rest are 0 (null effects)

intercept = 10.0

coef_df = pd.DataFrame({
    'Variable': [f'X{i+1}' for i in range(p)],
    'True Coefficient': true_beta
})
st.dataframe(coef_df, hide_index=True)

st.write(f"""
**Justification:**
- X1, X2: Strong effects (β=5.0, 3.0) - main predictors
- X3, X4: Moderate effects (β=-2.5, 1.5) - secondary predictors
- X5: Weak effect (β=0.5) - marginal predictor
- X6-X10: No effect (β=0) - noise variables
- Noise: σ={noise_std} controls error variance
""")

# Generate data
np.random.seed(seed)
X = np.random.randn(n, p)
y = intercept + X @ true_beta + np.random.randn(n) * noise_std

df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(p)])
df['y'] = y

# Fit model
st.header("2. Linear Regression Results")
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate statistics
residuals = y - y_pred
n_samples, n_features = X.shape
dof = n_samples - n_features - 1
mse = np.sum(residuals**2) / dof
var_beta = mse * np.linalg.inv(X.T @ X).diagonal()
se_beta = np.sqrt(var_beta)
t_stats = model.coef_ / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

results_df = pd.DataFrame({
    'Variable': [f'X{i+1}' for i in range(p)],
    'True β': true_beta,
    'Estimated β': model.coef_,
    'Std Error': se_beta,
    't-statistic': t_stats,
    'p-value': p_values,
    'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' for p in p_values]
})

st.dataframe(results_df.style.format({
    'True β': '{:.3f}',
    'Estimated β': '{:.3f}',
    'Std Error': '{:.3f}',
    't-statistic': '{:.3f}',
    'p-value': '{:.4f}'
}), hide_index=True)

st.write(f"**Intercept:** {model.intercept_:.3f} (True: {intercept:.1f})")
st.write(f"**R² Score:** {model.score(X, y):.3f}")

# Significance analysis
st.header("3. Statistical Significance Analysis")
sig_vars = results_df[results_df['p-value'] < 0.05]['Variable'].tolist()
st.write(f"**Significant variables (p < 0.05):** {', '.join(sig_vars)}")

true_nonzero = [f'X{i+1}' for i in range(p) if true_beta[i] != 0]
st.write(f"**True non-zero coefficients:** {', '.join(true_nonzero)}")

match = set(sig_vars) == set(true_nonzero)
st.write(f"**Do they match?** {'✓ Yes' if match else '✗ No'}")

# Regression diagnostics
st.header("4. Regression Diagnostics")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# Scale-Location
standardized_residuals = residuals / np.std(residuals)
axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Standardized residuals|')
axes[1, 0].set_title('Scale-Location')

# Residuals histogram
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Residuals Distribution')

plt.tight_layout()
st.pyplot(fig)

# High leverage points
st.header("5. High Leverage Points")

# Calculate leverage
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = H.diagonal()
leverage_threshold = 2 * (p + 1) / n

high_leverage_idx = np.where(leverage > leverage_threshold)[0]
st.write(f"**Leverage threshold:** {leverage_threshold:.4f}")
st.write(f"**Number of high leverage points:** {len(high_leverage_idx)}")

# Cook's distance
cooks_d = (standardized_residuals**2 / p) * (leverage / (1 - leverage)**2)
cooks_threshold = 4 / n

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

# Leverage plot
axes2[0].scatter(range(n), leverage, alpha=0.5)
axes2[0].axhline(y=leverage_threshold, color='r', linestyle='--', label='Threshold')
axes2[0].set_xlabel('Observation Index')
axes2[0].set_ylabel('Leverage')
axes2[0].set_title('Leverage Values')
axes2[0].legend()

# Cook's distance
axes2[1].scatter(range(n), cooks_d, alpha=0.5)
axes2[1].axhline(y=cooks_threshold, color='r', linestyle='--', label='Threshold')
axes2[1].set_xlabel('Observation Index')
axes2[1].set_ylabel("Cook's Distance")
axes2[1].set_title("Cook's Distance")
axes2[1].legend()

plt.tight_layout()
st.pyplot(fig2)

if len(high_leverage_idx) > 0:
    st.write(f"**High leverage observations:** {high_leverage_idx[:10].tolist()}" + 
             (f" ... ({len(high_leverage_idx)} total)" if len(high_leverage_idx) > 10 else ""))

influential_idx = np.where(cooks_d > cooks_threshold)[0]
if len(influential_idx) > 0:
    st.write(f"**Influential points (Cook's D > {cooks_threshold:.4f}):** {influential_idx.tolist()}")
else:
    st.write("**No influential points detected**")