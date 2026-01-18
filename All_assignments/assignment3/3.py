import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

st.title("Non-Linear Regression and Bias-Variance Trade-off")

# Sidebar
st.sidebar.header("Parameters")
function_type = st.sidebar.selectbox(
    "Generating Function",
    ["Polynomial", "Exponential + Sine", "Sine + Cosine", "Polynomial + Sine"]
)
seed = st.sidebar.number_input("Random seed", value=42, min_value=0)
noise_level = st.sidebar.slider("Noise level", 0.0, 2.0, 0.5, 0.1)

# Define true function
def true_function(x, func_type):
    if func_type == "Polynomial":
        return 2*x**3 - 3*x**2 + x + 1
    elif func_type == "Exponential + Sine":
        return np.exp(-x) * np.sin(2*np.pi*x) + 0.5*x
    elif func_type == "Sine + Cosine":
        return np.sin(2*x) + 0.5*np.cos(4*x)
    else:  # Polynomial + Sine
        return x**2 + 0.5*np.sin(4*x)

# Function formula
function_formulas = {
    "Polynomial": r"f(x) = 2x^3 - 3x^2 + x + 1",
    "Exponential + Sine": r"f(x) = e^{-x} \sin(2\pi x) + 0.5x",
    "Sine + Cosine": r"f(x) = \sin(2x) + 0.5\cos(4x)",
    "Polynomial + Sine": r"f(x) = x^2 + 0.5\sin(4x)"
}

st.header("1. Small Dataset (n=10)")
st.write("**True Generating Function:**")
st.latex(function_formulas[function_type])

# Generate small dataset
np.random.seed(seed)
n_small = 10
X_small = np.sort(np.random.uniform(-2, 2, n_small))
y_true_small = true_function(X_small, function_type)
y_small = y_true_small + np.random.randn(n_small) * noise_level

# For plotting
X_plot = np.linspace(-2, 2, 200)
y_plot_true = true_function(X_plot, function_type)

# Fit models
degree_low = st.sidebar.slider("Lower degree polynomial", 1, 9, 3)

model_low = make_pipeline(PolynomialFeatures(degree_low), LinearRegression())
model_high = make_pipeline(PolynomialFeatures(10), LinearRegression())

model_low.fit(X_small.reshape(-1, 1), y_small)
model_high.fit(X_small.reshape(-1, 1), y_small)

y_pred_low = model_low.predict(X_plot.reshape(-1, 1))
y_pred_high = model_high.predict(X_plot.reshape(-1, 1))

# Plot
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.scatter(X_small, y_small, color='black', s=100, zorder=5, label='Observed data (n=10)')
ax1.plot(X_plot, y_plot_true, 'g-', linewidth=2, label='True function')
ax1.plot(X_plot, y_pred_low, 'b--', linewidth=2, label=f'Polynomial degree {degree_low}')
ax1.plot(X_plot, y_pred_high, 'r-', linewidth=2, label='Polynomial degree 10')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'Polynomial Regression on Small Dataset (n=10)')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(min(y_small.min(), y_plot_true.min()) - 1, max(y_small.max(), y_plot_true.max()) + 1)
st.pyplot(fig1)

# Metrics
train_mse_low = mean_squared_error(y_small, model_low.predict(X_small.reshape(-1, 1)))
train_mse_high = mean_squared_error(y_small, model_high.predict(X_small.reshape(-1, 1)))
train_r2_low = r2_score(y_small, model_low.predict(X_small.reshape(-1, 1)))
train_r2_high = r2_score(y_small, model_high.predict(X_small.reshape(-1, 1)))

metrics_small = pd.DataFrame({
    'Model': [f'Polynomial degree {degree_low}', 'Polynomial degree 10'],
    'Train MSE': [train_mse_low, train_mse_high],
    'Train R²': [train_r2_low, train_r2_high]
})

st.dataframe(metrics_small.style.format({
    'Train MSE': '{:.6f}',
    'Train R²': '{:.6f}'
}), hide_index=True)

st.write("""
**Observations:**
- **Lowest train error:** Degree 10 polynomial (MSE ≈ 0, perfect fit)
- **Closer to true function:** Lower degree polynomial (less overfitting)
- **Why degree 10 interpolates all points:** With 10 data points and 11 parameters (degree 10 polynomial has 11 coefficients including intercept), the model has enough flexibility to pass through every single point exactly
- **Bias-Variance Trade-off:**
  - **Lower degree (high bias, low variance):** Underfits training data but generalizes better
  - **Degree 10 (low bias, high variance):** Perfectly fits training data but fails to generalize, capturing noise instead of signal
""")

# Large dataset
st.header("2. Large Dataset (n>100)")

n_large = st.sidebar.slider("Large dataset size", 100, 500, 200, 50)

np.random.seed(seed)
X_large = np.sort(np.random.uniform(-2, 2, n_large))
y_true_large = true_function(X_large, function_type)
y_large = y_true_large + np.random.randn(n_large) * noise_level

# Fit degree 10 on large dataset
model_high_large = make_pipeline(PolynomialFeatures(10), LinearRegression())
model_high_large.fit(X_large.reshape(-1, 1), y_large)
y_pred_high_large = model_high_large.predict(X_plot.reshape(-1, 1))

# Plot comparison
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

# Small dataset
ax2a.scatter(X_small, y_small, color='black', s=100, zorder=5, alpha=0.6)
ax2a.plot(X_plot, y_plot_true, 'g-', linewidth=2, label='True function')
ax2a.plot(X_plot, y_pred_high, 'r-', linewidth=2, label='Degree 10 (n=10)')
ax2a.set_xlabel('x')
ax2a.set_ylabel('y')
ax2a.set_title(f'Degree 10 Polynomial - Small Dataset (n=10)')
ax2a.legend()
ax2a.grid(alpha=0.3)

# Large dataset
ax2b.scatter(X_large, y_large, color='black', s=20, zorder=5, alpha=0.3)
ax2b.plot(X_plot, y_plot_true, 'g-', linewidth=2, label='True function')
ax2b.plot(X_plot, y_pred_high_large, 'b-', linewidth=2, label=f'Degree 10 (n={n_large})')
ax2b.set_xlabel('x')
ax2b.set_ylabel('y')
ax2b.set_title(f'Degree 10 Polynomial - Large Dataset (n={n_large})')
ax2b.legend()
ax2b.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)

train_mse_high_large = mean_squared_error(y_large, model_high_large.predict(X_large.reshape(-1, 1)))
train_r2_high_large = r2_score(y_large, model_high_large.predict(X_large.reshape(-1, 1)))

st.write(f"""
**Comparison:**
- **Small dataset (n=10):** Extreme overfitting, erratic predictions between points
- **Large dataset (n={n_large}):** Much smoother curve, follows true function better
- **Why?** With more data points, the model has more constraints and cannot fit noise as easily. The ratio of parameters to data points decreases (11 params / {n_large} points vs 11/10), reducing overfitting
""")

# Non-linear models
st.header("3. Non-Linear Regression Models")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

models = {
    'Polynomial (degree 3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
    'Polynomial (degree 10)': make_pipeline(PolynomialFeatures(10), LinearRegression()),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=seed),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=seed),
    'SVR (RBF kernel)': SVR(kernel='rbf', C=100, gamma=0.1),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=10)
}

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
axes3 = axes3.ravel()

results = []

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_large.reshape(-1, 1), y_large)
    y_pred = model.predict(X_plot.reshape(-1, 1))
    
    train_pred = model.predict(X_large.reshape(-1, 1))
    mse = mean_squared_error(y_large, train_pred)
    r2 = r2_score(y_large, train_pred)
    
    results.append({'Model': name, 'MSE': mse, 'R²': r2})
    
    axes3[idx].scatter(X_large, y_large, color='black', s=10, alpha=0.3, zorder=3)
    axes3[idx].plot(X_plot, y_plot_true, 'g-', linewidth=2, label='True', zorder=4)
    axes3[idx].plot(X_plot, y_pred, 'r-', linewidth=2, label='Predicted', zorder=5)
    axes3[idx].set_title(f'{name}\nMSE={mse:.4f}, R²={r2:.4f}')
    axes3[idx].set_xlabel('x')
    axes3[idx].set_ylabel('y')
    axes3[idx].legend(fontsize=8)
    axes3[idx].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig3)

results_df = pd.DataFrame(results).sort_values('MSE')
st.subheader("Model Performance Comparison")
st.dataframe(results_df.style.format({
    'MSE': '{:.6f}',
    'R²': '{:.6f}'
}), hide_index=True)

st.write("""
**Key Insights:**
- **Polynomial models:** Simple but can overfit with high degree
- **Random Forest & Gradient Boosting:** Flexible, often perform well on complex patterns
- **SVR:** Good for smooth non-linear relationships, depends on kernel and hyperparameters
- **KNN:** Simple, local averaging, can capture complex patterns but sensitive to noise
""")