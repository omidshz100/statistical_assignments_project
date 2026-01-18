from functools import lru_cache
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from scipy import stats

# Page config
st.set_page_config(page_title="Assignment 3", layout="wide")

# Main title
st.title("Assignment 3: Regression Analysis")
st.markdown("---")

# Sidebar navigation
page = st.sidebar.radio(
    "Select a section:",
    ["Overview", "Part 1: Linear Regression & Diagnostics", 
     "Part 2: Regularization Methods", "Part 3: Non-Linear Regression"]
)

# Overview page
if page == "Overview":
    st.header("Assignment 3: Comprehensive Regression Analysis")
    st.markdown("""
    This assignment explores different aspects of regression modeling through three integrated parts:
    
    ### 📊 Part 1: Linear Regression & Diagnostics
    - Hypothesis testing and statistical significance
    - Regression diagnostics (residual plots, Q-Q plots, etc.)
    - Leverage analysis and influential points detection
    - Coefficient estimation with standard errors and p-values
    
    ### 🎯 Part 2: Regularization Methods
    - Ridge, Lasso, and Elastic Net regression
    - Hyperparameter tuning via cross-validation
    - Feature selection through regularization
    - Coefficient shrinkage visualization
    
    ### 📈 Part 3: Non-Linear Regression & Bias-Variance Trade-off
    - Polynomial regression with varying degrees
    - Multiple non-linear models (Random Forest, Gradient Boosting, SVR, KNN)
    - Overfitting vs. generalization on small vs. large datasets
    - Model comparison and performance metrics
    
    **Use the sidebar to navigate between sections!**
    """)

# ==================== PART 1: LINEAR REGRESSION ====================
elif page == "Part 1: Linear Regression & Diagnostics":
    st.header("1. Linear Regression Simulation and Diagnostics")
    
    # Sidebar for parameters
    st.sidebar.markdown("---")
    st.sidebar.header("Part 1: Parameters")
    n = st.sidebar.number_input("Sample size (n)", value=300, min_value=50, max_value=1000, key="p1_n")
    p = st.sidebar.number_input("Number of predictors (p)", value=10, min_value=2, max_value=20, key="p1_p")
    noise_std = st.sidebar.slider("Noise std deviation", 0.1, 5.0, 1.0, 0.1, key="p1_noise")
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0, key="p1_seed")
    
    # True coefficients
    st.subheader("1.1 Data Generation")
    st.write("**True Model:**")
    st.latex(r"y = \beta_0 + \sum_{i=1}^{p} \beta_i x_i + \epsilon, \quad \epsilon \sim N(0, \sigma^2)")
    
    true_beta = np.zeros(p)
    true_beta[0] = 5.0
    true_beta[1] = 3.0
    true_beta[2] = -2.5
    true_beta[3] = 1.5
    true_beta[4] = 0.5
    
    intercept = 10.0
    
    coef_df = pd.DataFrame({
        'Variable': [f'X{i+1}' for i in range(p)],
        'True Coefficient': true_beta
    })
    st.dataframe(coef_df, hide_index=True)
    
    # Generate data
    @st.cache_data
    def generate_data(n, p, noise_std, seed):
        np.random.seed(seed)
        X = np.random.randn(n, p)
        y = intercept + X @ true_beta + np.random.randn(n) * noise_std
        return X, y

    X, y = generate_data(n, p, noise_std, seed)
    
    # Fit model
    st.subheader("1.2 Linear Regression Results")
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
    st.subheader("1.3 Statistical Significance Analysis")
    sig_vars = results_df[results_df['p-value'] < 0.05]['Variable'].tolist()
    st.write(f"**Significant variables (p < 0.05):** {', '.join(sig_vars)}")
    
    true_nonzero = [f'X{i+1}' for i in range(p) if true_beta[i] != 0]
    st.write(f"**True non-zero coefficients:** {', '.join(true_nonzero)}")
    
    match = set(sig_vars) == set(true_nonzero)
    st.write(f"**Do they match?** {'✓ Yes' if match else '✗ No'}")
    
    # Regression diagnostics
    st.subheader("1.4 Regression Diagnostics")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 0].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized residuals|')
    axes[1, 0].set_title('Scale-Location')
    
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residuals Distribution')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # High leverage points
    st.subheader("1.5 High Leverage Points & Influential Points")
    
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = H.diagonal()
    leverage_threshold = 2 * (p + 1) / n
    
    high_leverage_idx = np.where(leverage > leverage_threshold)[0]
    st.write(f"**Leverage threshold:** {leverage_threshold:.4f}")
    st.write(f"**Number of high leverage points:** {len(high_leverage_idx)}")
    
    cooks_d = (standardized_residuals**2 / p) * (leverage / (1 - leverage)**2)
    cooks_threshold = 4 / n
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    
    axes2[0].scatter(range(n), leverage, alpha=0.5)
    axes2[0].axhline(y=leverage_threshold, color='r', linestyle='--', label='Threshold')
    axes2[0].set_xlabel('Observation Index')
    axes2[0].set_ylabel('Leverage')
    axes2[0].set_title('Leverage Values')
    axes2[0].legend()
    
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

# ==================== PART 2: REGULARIZATION ====================
elif page == "Part 2: Regularization Methods":
    st.header("2. Shrinkage and Regularization Methods")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Part 2: Parameters")
    n = st.sidebar.number_input("Sample size (n)", value=300, min_value=50, max_value=1000, key="p2_n")
    p = st.sidebar.number_input("Number of predictors (p)", value=10, min_value=2, max_value=20, key="p2_p")
    noise_std = st.sidebar.slider("Noise std deviation", 0.1, 5.0, 1.0, 0.1, key="p2_noise")
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0, key="p2_seed")
    
    st.subheader("2.0 Data Generation")
    true_beta = np.zeros(p)
    true_beta[0] = 5.0
    true_beta[1] = 3.0
    true_beta[2] = -2.5
    true_beta[3] = 1.5
    true_beta[4] = 0.5
    
    intercept = 10.0
    
    np.random.seed(seed)
    X = np.random.randn(n, p)
    y = intercept + X @ true_beta + np.random.randn(n) * noise_std
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.write(f"**True non-zero coefficients:** X1={true_beta[0]}, X2={true_beta[1]}, X3={true_beta[2]}, X4={true_beta[3]}, X5={true_beta[4]}")
    
    # Ridge Regression
    st.subheader("2.1 Ridge Regression")
    
    alphas_ridge = np.logspace(-3, 3, 100)
    ridge_cv = RidgeCV(alphas=alphas_ridge, cv=10)
    ridge_cv.fit(X_scaled, y)
    best_alpha_ridge = ridge_cv.alpha_
    
    st.write(f"**Optimal λ (alpha):** {best_alpha_ridge:.4f}")
    
    coefs_ridge = []
    for alpha in alphas_ridge:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        coefs_ridge.append(ridge.coef_)
    
    coefs_ridge = np.array(coefs_ridge)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i in range(p):
        ax1.plot(alphas_ridge, coefs_ridge[:, i], label=f'X{i+1}')
    ax1.axvline(float(best_alpha_ridge), color='red', linestyle='--', label='Optimal λ')
    ax1.set_xscale('log')
    ax1.set_xlabel('λ (alpha)')
    ax1.set_ylabel('Coefficient values')
    ax1.set_title('Ridge Regression: Coefficient Paths')
    ax1.legend(loc='best', ncol=2)
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    
    ridge_final = Ridge(alpha=best_alpha_ridge)
    ridge_final.fit(X_scaled, y)
    
    ridge_comparison = pd.DataFrame({
        'Variable': [f'X{i+1}' for i in range(p)],
        'True β': true_beta,
        'Ridge β': ridge_final.coef_,
        'Difference': np.abs(true_beta - ridge_final.coef_)
    })
    
    st.dataframe(ridge_comparison.style.format({
        'True β': '{:.3f}',
        'Ridge β': '{:.3f}',
        'Difference': '{:.3f}'
    }), hide_index=True)
    
    # Lasso Regression
    st.subheader("2.2 Lasso Regression")
    
    alphas_lasso = np.logspace(-3, 1, 100)
    lasso_cv = LassoCV(alphas=alphas_lasso, cv=10, max_iter=10000)
    lasso_cv.fit(X_scaled, y)
    best_alpha_lasso = lasso_cv.alpha_
    
    st.write(f"**Optimal λ (alpha):** {best_alpha_lasso:.4f}")
    
    coefs_lasso = []
    for alpha in alphas_lasso:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_scaled, y)
        coefs_lasso.append(lasso.coef_)
    
    coefs_lasso = np.array(coefs_lasso)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(p):
        ax2.plot(alphas_lasso, coefs_lasso[:, i], label=f'X{i+1}')
    ax2.axvline(float(best_alpha_lasso), color='red', linestyle='--', label='Optimal λ')
    ax2.set_xscale('log')
    ax2.set_xlabel('λ (alpha)')
    ax2.set_ylabel('Coefficient values')
    ax2.set_title('Lasso Regression: Coefficient Paths')
    ax2.legend(loc='best', ncol=2)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    
    lasso_final = Lasso(alpha=best_alpha_lasso, max_iter=10000)
    lasso_final.fit(X_scaled, y)
    
    zero_coefs = [f'X{i+1}' for i in range(p) if abs(lasso_final.coef_[i]) < 1e-5]
    nonzero_coefs = [f'X{i+1}' for i in range(p) if abs(lasso_final.coef_[i]) >= 1e-5]
    true_nonzero = [f'X{i+1}' for i in range(p) if abs(true_beta[i]) > 0]
    
    st.write(f"**Coefficients shrunk to zero:** {', '.join(zero_coefs) if zero_coefs else 'None'}")
    st.write(f"**Non-zero coefficients:** {', '.join(nonzero_coefs)}")
    st.write(f"**True non-zero:** {', '.join(true_nonzero)}")
    
    lasso_comparison = pd.DataFrame({
        'Variable': [f'X{i+1}' for i in range(p)],
        'True β': true_beta,
        'Lasso β': lasso_final.coef_,
        'Exact Zero': [abs(lasso_final.coef_[i]) < 1e-5 for i in range(p)]
    })
    
    st.dataframe(lasso_comparison.style.format({
        'True β': '{:.3f}',
        'Lasso β': '{:.3f}'
    }), hide_index=True)
    
    # Elastic Net
    st.subheader("2.3 Elastic Net Regression")
    
    l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
    elastic_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas_lasso, cv=10, max_iter=10000)
    elastic_cv.fit(X_scaled, y)
    best_alpha_elastic = elastic_cv.alpha_
    best_l1_ratio = elastic_cv.l1_ratio_
    
    st.write(f"**Optimal λ (alpha):** {best_alpha_elastic:.4f}")
    st.write(f"**Optimal l1_ratio:** {best_l1_ratio:.4f}")
    st.write(f"*l1_ratio=1 is Lasso, l1_ratio=0 is Ridge*")
    
    st.write("**Cross-Validation Scores Heatmap**")
    
    cv_scores = np.zeros((len(l1_ratios), len(alphas_lasso[:20])))
    for i, l1_ratio in enumerate(l1_ratios):
        for j, alpha in enumerate(alphas_lasso[:20]):
            elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            scores = cross_val_score(elastic, X_scaled, y, cv=5, scoring='r2')
            cv_scores[i, j] = scores.mean()
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    im = ax3.imshow(cv_scores, aspect='auto', cmap='viridis')
    ax3.set_xlabel('λ index (log scale)')
    ax3.set_ylabel('l1_ratio')
    ax3.set_yticks(range(len(l1_ratios)))
    ax3.set_yticklabels([f'{r:.2f}' for r in l1_ratios])
    ax3.set_title('Elastic Net: Cross-Validation R² Scores')
    plt.colorbar(im, ax=ax3, label='R² Score')
    st.pyplot(fig3)
    
    elastic_final = ElasticNet(alpha=best_alpha_elastic, l1_ratio=best_l1_ratio, max_iter=10000)
    elastic_final.fit(X_scaled, y)
    
    # Comparison
    st.subheader("2.4 Comparison of All Methods")
    
    comparison_df = pd.DataFrame({
        'Variable': [f'X{i+1}' for i in range(p)],
        'True β': true_beta,
        'Ridge β': ridge_final.coef_,
        'Lasso β': lasso_final.coef_,
        'ElasticNet β': elastic_final.coef_
    })
    
    st.dataframe(comparison_df.style.format({
        'True β': '{:.3f}',
        'Ridge β': '{:.3f}',
        'Lasso β': '{:.3f}',
        'ElasticNet β': '{:.3f}'
    }), hide_index=True)
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(p)
    width = 0.2
    
    ax4.bar(x_pos - 1.5*width, true_beta, width, label='True', alpha=0.8)
    ax4.bar(x_pos - 0.5*width, ridge_final.coef_, width, label='Ridge', alpha=0.8)
    ax4.bar(x_pos + 0.5*width, lasso_final.coef_, width, label='Lasso', alpha=0.8)
    ax4.bar(x_pos + 1.5*width, elastic_final.coef_, width, label='ElasticNet', alpha=0.8)
    
    ax4.set_xlabel('Variables')
    ax4.set_ylabel('Coefficient values')
    ax4.set_title('Comparison of Coefficient Estimates')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'X{i+1}' for i in range(p)])
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    st.pyplot(fig4)

# ==================== PART 3: NON-LINEAR REGRESSION ====================
elif page == "Part 3: Non-Linear Regression & Bias-Variance Trade-off":
    st.header("3. Non-Linear Regression and Bias-Variance Trade-off")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Part 3: Parameters")
    function_type = st.sidebar.selectbox(
        "Generating Function",
        ["Polynomial", "Exponential + Sine", "Sine + Cosine", "Polynomial + Sine"],
        key="p3_func"
    )
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0, key="p3_seed")
    noise_level = st.sidebar.slider("Noise level", 0.0, 2.0, 0.5, 0.1, key="p3_noise")
    
    # Define true function
    def true_function(x, func_type):
        if func_type == "Polynomial":
            return 2*x**3 - 3*x**2 + x + 1
        elif func_type == "Exponential + Sine":
            return np.exp(-x) * np.sin(2*np.pi*x) + 0.5*x
        elif func_type == "Sine + Cosine":
            return np.sin(2*x) + 0.5*np.cos(4*x)
        else:
            return x**2 + 0.5*np.sin(4*x)
    
    function_formulas = {
        "Polynomial": r"f(x) = 2x^3 - 3x^2 + x + 1",
        "Exponential + Sine": r"f(x) = e^{-x} \sin(2\pi x) + 0.5x",
        "Sine + Cosine": r"f(x) = \sin(2x) + 0.5\cos(4x)",
        "Polynomial + Sine": r"f(x) = x^2 + 0.5\sin(4x)"
    }
    
    st.subheader("3.1 Small Dataset (n=10)")
    st.write("**True Generating Function:**")
    st.latex(function_formulas[function_type])
    
    np.random.seed(seed)
    n_small = 10
    X_small = np.sort(np.random.uniform(-2, 2, n_small))
    y_true_small = true_function(X_small, function_type)
    y_small = y_true_small + np.random.randn(n_small) * noise_level
    
    X_plot = np.linspace(-2, 2, 200)
    y_plot_true = true_function(X_plot, function_type)
    
    degree_low = st.sidebar.slider("Lower degree polynomial", 1, 9, 3, key="p3_deg_low")
    
    model_low = make_pipeline(PolynomialFeatures(degree_low), LinearRegression())
    model_high = make_pipeline(PolynomialFeatures(10), LinearRegression())
    
    model_low.fit(X_small.reshape(-1, 1), y_small)
    model_high.fit(X_small.reshape(-1, 1), y_small)
    
    y_pred_low = model_low.predict(X_plot.reshape(-1, 1))
    y_pred_high = model_high.predict(X_plot.reshape(-1, 1))
    
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
    
    st.subheader("3.2 Large Dataset (n>100)")
    
    n_large = st.sidebar.slider("Large dataset size", 100, 500, 200, 50, key="p3_n_large")
    
    np.random.seed(seed)
    X_large = np.sort(np.random.uniform(-2, 2, n_large))
    y_true_large = true_function(X_large, function_type)
    y_large = y_true_large + np.random.randn(n_large) * noise_level
    
    model_high_large = make_pipeline(PolynomialFeatures(10), LinearRegression())
    model_high_large.fit(X_large.reshape(-1, 1), y_large)
    y_pred_high_large = model_high_large.predict(X_plot.reshape(-1, 1))
    
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax2a.scatter(X_small, y_small, color='black', s=100, zorder=5, alpha=0.6)
    ax2a.plot(X_plot, y_plot_true, 'g-', linewidth=2, label='True function')
    ax2a.plot(X_plot, y_pred_high, 'r-', linewidth=2, label='Degree 10 (n=10)')
    ax2a.set_xlabel('x')
    ax2a.set_ylabel('y')
    ax2a.set_title(f'Degree 10 Polynomial - Small Dataset (n=10)')
    ax2a.legend()
    ax2a.grid(alpha=0.3)
    
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
    
    st.subheader("3.3 Non-Linear Regression Models")
    
    models = {
        'Polynomial (degree 3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
        'Polynomial (degree 10)': make_pipeline(PolynomialFeatures(10), LinearRegression()),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=int(seed)),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=int(seed)),
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
    st.dataframe(results_df.style.format({
        'MSE': '{:.6f}',
        'R²': '{:.6f}'
    }), hide_index=True)