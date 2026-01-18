import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    roc_auc_score,
    accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title
st.title("🎓 End-to-End Supervised Classification Project")
st.subheader("Customer Churn Prediction - Statistical Learning Course")

# Sidebar for model selection and hyperparameters
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Random Forest", "Support Vector Machine (SVM)"]
)

# Model-specific hyperparameters
st.sidebar.subheader("Hyperparameters")
if model_choice == "Logistic Regression":
    C_param = st.sidebar.slider("Regularization Parameter (C)", 0.01, 10.0, 1.0, 0.1)
    max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200, 100)
elif model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100, 10)
    max_depth = st.sidebar.slider("Max Depth", 2, 20, 10, 1)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, 1)
else:  # SVM
    C_param = st.sidebar.slider("Regularization Parameter (C)", 0.1, 10.0, 1.0, 0.1)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

# Generate synthetic dataset
@st.cache_data
def generate_dataset(n_samples=1000, random_state=42):
    """
    Generate a synthetic customer churn dataset.
    Features represent typical customer attributes.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],  # Imbalanced classes (70% not churned, 30% churned)
        flip_y=0.05,  # Add 5% label noise for realism
        random_state=random_state
    )
    
    # Create meaningful feature names
    feature_names = [
        'Age',
        'MonthlyCharge',
        'CustomerTenure',
        'UsageScore',
        'SupportTickets',
        'ContractLength',
        'PaymentDelay',
        'ServiceRating'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Churn'] = y
    
    # Scale features to realistic ranges
    df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min()) * 50 + 20  # 20-70
    df['MonthlyCharge'] = (df['MonthlyCharge'] - df['MonthlyCharge'].min()) / (df['MonthlyCharge'].max() - df['MonthlyCharge'].min()) * 100 + 20  # 20-120
    df['CustomerTenure'] = (df['CustomerTenure'] - df['CustomerTenure'].min()) / (df['CustomerTenure'].max() - df['CustomerTenure'].min()) * 60  # 0-60 months
    df['UsageScore'] = (df['UsageScore'] - df['UsageScore'].min()) / (df['UsageScore'].max() - df['UsageScore'].min()) * 100  # 0-100
    df['SupportTickets'] = (df['SupportTickets'] - df['SupportTickets'].min()) / (df['SupportTickets'].max() - df['SupportTickets'].min()) * 20  # 0-20
    df['ContractLength'] = (df['ContractLength'] - df['ContractLength'].min()) / (df['ContractLength'].max() - df['ContractLength'].min()) * 36  # 0-36 months
    df['PaymentDelay'] = (df['PaymentDelay'] - df['PaymentDelay'].min()) / (df['PaymentDelay'].max() - df['PaymentDelay'].min()) * 30  # 0-30 days
    df['ServiceRating'] = (df['ServiceRating'] - df['ServiceRating'].min()) / (df['ServiceRating'].max() - df['ServiceRating'].min()) * 5  # 0-5 stars
    
    return df, feature_names

# Load data
df, feature_names = generate_dataset()
X = df[feature_names]
y = df['Churn']

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Problem Definition",
    "📊 EDA",
    "🔧 Preprocessing & PCA",
    "🤖 Model Building",
    "📈 Evaluation"
])

# ============================================================
# TAB 1: MOTIVATION & PROBLEM DEFINITION
# ============================================================
with tab1:
    st.header("1. Motivation & Problem Definition")
    
    st.markdown("""
    ### 🎯 Business Context
    **Customer Churn** refers to when customers stop doing business with a company. 
    For subscription-based businesses (telecom, SaaS, streaming services), predicting 
    which customers are likely to churn is critical for:
    
    - **Retention Strategies**: Proactively engage at-risk customers
    - **Revenue Protection**: Reduce customer acquisition costs
    - **Resource Optimization**: Focus marketing efforts on high-risk segments
    
    ### 📝 Problem Statement
    Build a **binary classification model** to predict whether a customer will churn (1) 
    or stay (0) based on their behavioral and demographic features.
    
    ### 🎓 Learning Objectives
    - Apply the complete machine learning pipeline
    - Understand preprocessing and feature engineering
    - Compare multiple classification algorithms
    - Interpret model performance using appropriate metrics
    """)
    
    st.subheader("Dataset Preview")
    st.markdown("""
    Our dataset contains **1000 customers** with 8 features representing customer 
    attributes and behavior. Let's examine the first few rows:
    """)
    st.dataframe(df.head(), use_container_width=True)
    
    st.info(f"**Dataset Shape**: {df.shape[0]} samples × {df.shape[1]} features (including target)")

# ============================================================
# TAB 2: EXPLORATORY DATA ANALYSIS
# ============================================================
with tab2:
    st.header("2. Exploratory Data Analysis (EDA)")
    
    st.markdown("""
    ### 📊 Why EDA?
    Before building models, we must **understand our data**:
    - Check for missing values and outliers
    - Understand feature distributions
    - Identify relationships between features
    - Detect class imbalance
    """)
    
    # Summary Statistics
    st.subheader("2.1 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Class Balance
    st.subheader("2.2 Class Balance Analysis")
    st.markdown("""
    **Class imbalance** can bias models toward the majority class. Let's check our target distribution:
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        churn_counts = df['Churn'].value_counts()
        st.metric("Not Churned (0)", churn_counts[0])
        st.metric("Churned (1)", churn_counts[1])
        st.metric("Imbalance Ratio", f"{churn_counts[0]/churn_counts[1]:.2f}:1")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x='Churn', palette='Set2', ax=ax)
        ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Churn (0 = No, 1 = Yes)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        for container in ax.containers:
            ax.bar_label(container) 
        st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("2.3 Correlation Heatmap")
    st.markdown("""
    **Correlation analysis** helps identify:
    - Which features are most related to churn
    - Multicollinearity between features (which can affect some models)
    """)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    st.pyplot(fig)
    
    # Feature Distributions
    st.subheader("2.4 Feature Distributions by Churn Status")
    st.markdown("""
    **Box plots** reveal how feature distributions differ between churned and non-churned customers:
    """)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx, feature in enumerate(feature_names):
        sns.boxplot(data=df, x='Churn', y=feature, palette='Set1', ax=axes[idx])
        axes[idx].set_title(f'{feature} by Churn', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Churn')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================================
# TAB 3: PREPROCESSING & UNSUPERVISED ANALYSIS
# ============================================================
with tab3:
    st.header("3. Preprocessing & Unsupervised Analysis")
    
    st.markdown("""
    ### 🔧 Why Preprocessing?
    Most machine learning algorithms require:
    - **Scaled features**: Algorithms like SVM and Logistic Regression are sensitive to feature magnitudes
    - **Train/Test split**: To evaluate generalization on unseen data
    """)
    
    # Train/Test Split
    st.subheader("3.1 Train/Test Split")
    test_size = st.slider("Select Test Set Size (%)", 10, 40, 20, 5) / 100
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", len(X_train))
    col2.metric("Test Samples", len(X_test))
    col3.metric("Split Ratio", f"{int((1-test_size)*100)}:{int(test_size*100)}")
    
    st.markdown("""
    ⚠️ **Stratified split** preserves class distribution in both train and test sets.
    """)
    
    # Feature Scaling
    st.subheader("3.2 Feature Scaling (Standardization)")
    st.markdown("""
    **StandardScaler** transforms features to have:
    - Mean = 0
    - Standard Deviation = 1
    
    Formula: $z = \\frac{x - \\mu}{\\sigma}$
    """)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.success("✅ Features scaled successfully!")
    
    # PCA Analysis
    st.subheader("3.3 Unsupervised Analysis: Principal Component Analysis (PCA)")
    st.markdown("""
    ### 🎨 Why PCA?
    **PCA** is a dimensionality reduction technique that:
    - Transforms high-dimensional data into lower dimensions
    - Preserves maximum variance
    - Helps visualize if classes are **linearly separable**
    
    We'll project our 8D data into 2D space to see if churned and non-churned 
    customers form distinct clusters.
    """)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_scaled)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Variance Explained (PC1)", f"{pca.explained_variance_ratio_[0]*100:.2f}%")
    with col2:
        st.metric("Variance Explained (PC2)", f"{pca.explained_variance_ratio_[1]*100:.2f}%")
    
    st.info(f"**Total Variance Captured**: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # PCA Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, 
                        cmap='RdYlGn_r', alpha=0.6, edgecolors='k', s=50)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('PCA: Customer Churn in 2D Space', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Churn (0=No, 1=Yes)', fontsize=10)
    st.pyplot(fig)
    
    st.markdown("""
    ### 📖 Interpretation:
    - **Overlapping clusters** → Classes are not linearly separable (complex decision boundary needed)
    - **Distinct clusters** → Linear models may perform well
    - PCA is useful for **visualization** and **noise reduction**, but we'll use original features for modeling
    """)

# ============================================================
# TAB 4: MODEL BUILDING & TUNING
# ============================================================
with tab4:
    st.header("4. Model Building & Training")
    
    st.markdown(f"""
    ### 🤖 Selected Model: **{model_choice}**
    
    Different algorithms have different strengths:
    - **Logistic Regression**: Fast, interpretable, works well for linearly separable data
    - **Random Forest**: Handles non-linear relationships, robust to outliers, provides feature importance
    - **SVM**: Effective in high-dimensional spaces, flexible with kernel functions
    """)
    
    # Build model based on selection
    if model_choice == "Logistic Regression":
        model = LogisticRegression(C=C_param, max_iter=max_iter, random_state=42)
        st.code(f"LogisticRegression(C={C_param}, max_iter={max_iter})")
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        st.code(f"RandomForestClassifier(n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split})")
    else:  # SVM
        model = SVC(C=C_param, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        st.code(f"SVC(C={C_param}, kernel='{kernel}', gamma='{gamma}', probability=True)")
    
    # Train model
    with st.spinner("🔄 Training model..."):
        model.fit(X_train_scaled, y_train)
    
    st.success("✅ Model trained successfully!")
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Get prediction probabilities (for ROC curve)
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_train_proba = model.decision_function(X_train_scaled)
        y_test_proba = model.decision_function(X_test_scaled)
    
    # Quick performance preview
    st.subheader("Quick Performance Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{accuracy_score(y_train, y_train_pred)*100:.2f}%")
    with col2:
        st.metric("Test Accuracy", f"{accuracy_score(y_test, y_test_pred)*100:.2f}%")
    
    if accuracy_score(y_train, y_train_pred) - accuracy_score(y_test, y_test_pred) > 0.1:
        st.warning("⚠️ Large gap between train and test accuracy may indicate overfitting!")

# ============================================================
# TAB 5: EVALUATION & INTERPRETATION
# ============================================================
with tab5:
    st.header("5. Model Evaluation & Interpretation")
    
    st.markdown("""
    ### 📊 Why Multiple Metrics?
    **Accuracy alone is misleading** for imbalanced datasets. We need:
    - **Confusion Matrix**: Shows types of errors (False Positives vs False Negatives)
    - **Precision**: Of predicted churns, how many actually churned?
    - **Recall**: Of actual churns, how many did we catch?
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Model's ability to distinguish between classes
    """)
    
    # Confusion Matrix
    st.subheader("5.1 Confusion Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Set**")
        cm_train = confusion_matrix(y_train, y_train_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_title('Training Set Confusion Matrix', fontsize=12, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Test Set**")
        cm_test = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold')
        st.pyplot(fig)
    
    st.markdown("""
    **📖 Reading the Matrix:**
    - **Top-Left (TN)**: Correctly predicted "No Churn"
    - **Top-Right (FP)**: Falsely predicted "Churn" (Type I error)
    - **Bottom-Left (FN)**: Missed actual "Churn" (Type II error) ⚠️ Most costly!
    - **Bottom-Right (TP)**: Correctly predicted "Churn" ✅
    """)
    
    # Classification Report
    st.subheader("5.2 Classification Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Set**")
        report_train = classification_report(y_train, y_train_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report_train).transpose(), use_container_width=True)
    
    with col2:
        st.markdown("**Test Set**")
        report_test = classification_report(y_test, y_test_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report_test).transpose(), use_container_width=True)
    
    st.markdown("""
    **📖 Metric Definitions:**
    - **Precision** = $\\frac{TP}{TP + FP}$ → "When we predict churn, how often are we right?"
    - **Recall (Sensitivity)** = $\\frac{TP}{TP + FN}$ → "Of all actual churns, how many did we catch?"
    - **F1-Score** = $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$ → Balanced metric
    - **Support** = Number of actual occurrences in the dataset
    """)
    
    # ROC Curve
    st.subheader("5.3 ROC Curve & AUC Score")
    st.markdown("""
    **ROC (Receiver Operating Characteristic)** curve shows the trade-off between:
    - **True Positive Rate (Recall)**: How many churns we catch
    - **False Positive Rate**: How many non-churns we misclassify
    
    **AUC (Area Under Curve)**: Overall model quality (0.5 = random, 1.0 = perfect)
    """)
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    auc_train = roc_auc_score(y_train, y_train_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr_train, tpr_train, label=f'Train (AUC = {auc_train:.3f})', linewidth=2, color='blue')
    ax.plot(fpr_test, tpr_test, label=f'Test (AUC = {auc_test:.3f})', linewidth=2, color='green')
    ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier (AUC = 0.5)', linewidth=2)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training AUC", f"{auc_train:.4f}")
    with col2:
        st.metric("Test AUC", f"{auc_test:.4f}")
    
    # Feature Importance (for tree-based models)
    if model_choice == "Random Forest":
        st.subheader("5.4 Feature Importance Analysis")
        st.markdown("""
        **Feature Importance** shows which features contribute most to predictions.
        Higher values indicate stronger influence on the model's decisions.
        """)
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis', ax=ax)
        ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        st.pyplot(fig)
        
        st.dataframe(feature_importance_df, use_container_width=True)
        
        st.markdown("""
        **🎯 Business Insight:**
        Focus retention efforts on the top features! For example, if `SupportTickets` 
        is most important, improving customer support could reduce churn.
        """)
    
    # Final Summary
    st.subheader("5.5 Summary & Recommendations")
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = report_test['1']['f1-score']
    
    st.markdown(f"""
    ### 🎓 Key Takeaways:
    
    **Model Performance:**
    - Test Accuracy: **{test_acc*100:.2f}%**
    - Test F1-Score (Churn Class): **{test_f1:.3f}**
    - Test AUC: **{auc_test:.3f}**
    
    **Student Learning Points:**
    1. ✅ Always split data before preprocessing to avoid data leakage
    2. ✅ Use stratified sampling to preserve class distribution
    3. ✅ Evaluate multiple metrics, not just accuracy
    4. ✅ Check for overfitting by comparing train vs test performance
    5. ✅ Interpret results in business context (cost of False Negatives vs False Positives)
    
    **Next Steps for Production:**
    - Cross-validation for more robust evaluation
    - Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
    - Handle class imbalance (SMOTE, class weights)
    - Feature engineering (interaction terms, polynomial features)
    - Model ensemble methods
    """)
    
    if test_acc > 0.85 and auc_test > 0.85:
        st.success("🎉 Excellent model performance! Ready for further validation.")
    elif test_acc > 0.75:
        st.info("✅ Good model performance. Consider hyperparameter tuning for improvement.")
    else:
        st.warning("⚠️ Model needs improvement. Try different algorithms or feature engineering.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>End-to-End Supervised Classification Project</b></p>
    <p>Statistical Learning Course | Master's Program in Data Science</p>
    <p><i>Built with Streamlit, scikit-learn, and ❤️</i></p>
</div>
""", unsafe_allow_html=True)