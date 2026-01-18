import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Statistical Learning Projects",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
with st.sidebar:
    st.title("📚 Navigation")
    st.markdown("### Select an Assignment:")
    st.info("Click on any page below to explore the analysis")

# Title and description
st.title("📊 Statistical Learning & Data Analysis")
st.markdown("""
## Master's Level Course Projects

Welcome to the **Statistical Learning and Data Analysis** course project portfolio. 
This interactive application showcases four comprehensive assignments covering:

- **📈 Assignment 1**: Exploratory Data Analysis (EDA)
- **📊 Assignment 2**: Probability Models Comparison
- **📉 Assignment 3**: Regression and Data Simulation
- **🤖 Assignment 4**: Classification & Final Project

""")

# Main content
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Project Goals
    
    Master key statistical learning techniques:
    - Data exploration and visualization
    - Probability models and distributions
    - Regression analysis and model diagnostics
    - Classification and machine learning
    - Dimensionality reduction (PCA)
    - Clustering algorithms
    """)

with col2:
    st.markdown("""
    ### 🛠️ Technologies Used
    
    - **Python** for analysis
    - **Pandas & NumPy** for data manipulation
    - **Scikit-learn** for ML models
    - **Matplotlib & Seaborn** for visualization
    - **Streamlit** for interactive interface
    """)

st.markdown("---")

st.markdown("""
### 📍 Navigation

**👈 Use the sidebar menu** to navigate between assignments. Each page includes:
- Complete analysis code
- Interactive visualizations
- Interpretative insights
- Statistical explanations
""")

# Assignment overview cards
st.markdown("### 📚 Assignments Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 📈 Assignment 1: Exploratory Data Analysis
    *Student habits & performance dataset analysis with PCA and clustering*
    
    #### 📊 Assignment 2: Probability Models
    *Comparing exponential, uniform, and normal distributions*
    
    #### 📉 Assignment 3.1: Linear Regression
    *Simple and multiple linear regression analysis*
    """)

with col2:
    st.markdown("""
    #### 📉 Assignment 3.2: Regularization
    *Ridge and Lasso regression techniques*
    
    #### 📉 Assignment 3.3: Non-Linear Models
    *Polynomial and non-linear regression*
    
    #### 🤖 Assignment 4: Classification
    *Machine learning classification models*
    """)

st.markdown("---")

st.info("💡 **Tip**: Click on any assignment in the sidebar (left) to start exploring!", icon="ℹ️")

st.markdown("---")

st.markdown("""
### 💡 Key Features

✅ **Interactive Exploration** - Adjust parameters and see results in real-time  
✅ **Professional Visualizations** - Publication-ready plots and charts  
✅ **Statistical Rigor** - Proper diagnostics and validation  
✅ **Interpretable Results** - Clear explanations of findings  
""")

# Footer
st.markdown("""
---
<div style="text-align: center">
    <p style="color: #888; font-size: 12px;">
        Built with Streamlit | Statistical Learning Course | 2026
    </p>
</div>
""", unsafe_allow_html=True)