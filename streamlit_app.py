import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Statistical Learning Projects",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.markdown("---")
st.markdown("### 📍 Navigate to Assignments")
st.markdown("Click on any assignment below to view the interactive analysis:")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("#### 📈 Assignment 1: Exploratory Data Analysis")
        st.markdown("*Student habits & performance dataset analysis with PCA and clustering*")
        st.markdown("[**📊 View Assignment 1 →**](/Assignment_1_EDA)")
    
    with st.container(border=True):
        st.markdown("#### 📊 Assignment 2: Probability Models")
        st.markdown("*Comparing exponential, uniform, and normal distributions*")
        st.markdown("[**🎲 View Assignment 2 →**](/Assignment_2_Probability)")
    
    with st.container(border=True):
        st.markdown("#### 📉 Assignment 3.1: Linear Regression")
        st.markdown("*Simple and multiple linear regression analysis*")
        st.markdown("[**📈 View Assignment 3.1 →**](/Assignment_3_Part_1_Linear_Regression)")

with col2:
    with st.container(border=True):
        st.markdown("#### 📉 Assignment 3.2: Regularization")
        st.markdown("*Ridge and Lasso regression techniques*")
        st.markdown("[**⚖️ View Assignment 3.2 →**](/Assignment_3_Part_2_Regularization)")
    
    with st.container(border=True):
        st.markdown("#### 📉 Assignment 3.3: Non-Linear Models")
        st.markdown("*Polynomial and non-linear regression*")
        st.markdown("[**🔄 View Assignment 3.3 →**](/Assignment_3_Part_3_NonLinear)")
    
    with st.container(border=True):
        st.markdown("#### 🤖 Assignment 4: Classification")
        st.markdown("*Machine learning classification models*")
        st.markdown("[**🎯 View Assignment 4 →**](/Assignment_4_Classification)")

st.markdown("---")
st.info("💡 **Note**: You can also use the sidebar menu (☰) in the top-left to navigate between pages!", icon="ℹ️")

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