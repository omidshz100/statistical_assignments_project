import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

st.title("Shrinkage and Regularization Methods")

# Sidebar for parameters
st.sidebar.header("Simulation Parameters")
n = st.sidebar.number_input("Sample size (n)", value=300, min_value=50, max_value=1000)
p = st.sidebar.number_input("Number of predictors (p)", value=10, min_value=2, max_value=20)
noise_std = st.sidebar.slider("Noise std deviation", 0.1, 5.0, 1.0, 0.1)
seed = st.sidebar.number_input("Random seed", value=42, min_value=0)

# Generate data
st.header("Data Generation")
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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.write(f"**True non-zero coefficients:** X1={true_beta[0]}, X2={true_beta[1]}, X3={true_beta[2]}, X4={true_beta[3]}, X5={true_beta[4]}")

# Ridge Regression
st.header("1. Ridge Regression")

alphas_ridge = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas_ridge, cv=10)
ridge_cv.fit(X_scaled, y)
best_alpha_ridge = ridge_cv.alpha_

st.write(f"**Optimal λ (alpha):** {best_alpha_ridge:.4f}")

# Coefficient paths for Ridge
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

# Ridge final coefficients
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
st.header("2. Lasso Regression")

alphas_lasso = np.logspace(-3, 1, 100)
lasso_cv = LassoCV(alphas=alphas_lasso, cv=10, max_iter=10000)
lasso_cv.fit(X_scaled, y)
best_alpha_lasso = lasso_cv.alpha_

st.write(f"**Optimal λ (alpha):** {best_alpha_lasso:.4f}")

# Coefficient paths for Lasso
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

# Lasso final coefficients
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

# Elastic Net Regression
st.header("3. Elastic Net Regression")

l1_ratios = [.1, .5, .7, .9, .95, .99, 1]
elastic_cv = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas_lasso, cv=10, max_iter=10000)
elastic_cv.fit(X_scaled, y)
best_alpha_elastic = elastic_cv.alpha_
best_l1_ratio = elastic_cv.l1_ratio_

st.write(f"**Optimal λ (alpha):** {best_alpha_elastic:.4f}")
st.write(f"**Optimal l1_ratio:** {best_l1_ratio:.4f}")
st.write(f"*l1_ratio=1 is Lasso, l1_ratio=0 is Ridge*")

# Grid search visualization
st.subheader("Cross-Validation Scores Heatmap")

cv_scores = np.zeros((len(l1_ratios), len(alphas_lasso[:20])))  # Use subset for speed
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

# Elastic Net final coefficients
elastic_final = ElasticNet(alpha=best_alpha_elastic, l1_ratio=best_l1_ratio, max_iter=10000)
elastic_final.fit(X_scaled, y)

# Comparison of all methods
st.header("4. Comparison of All Methods")

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

# Visualization
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