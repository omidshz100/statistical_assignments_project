# Statistical Learning and Data Analysis - Course Projects

A collection of four progressive assignments focusing on statistical learning, data analysis, and machine learning techniques. Each assignment builds upon foundational statistical concepts to develop comprehensive data analysis skills.

## Project Overview

This course consists of four major assignments that cover the full spectrum of statistical learning:

1. **Assignment 1: Exploratory Data Analysis (EDA)**
2. **Assignment 2: Probability Models Comparison**
3. **Assignment 3: Regression and Data Simulation**
4. **Assignment 4: Classification & Final Project**

---

## Assignment 1: Exploratory Data Analysis (EDA)

### Objective
Perform a complete exploratory data analysis on a selected dataset by answering structured questions that build upon each other.

### Deliverables
- Analysis file (`.ipynb` or `.Rmd`)
- Export formats: `.pdf` or `.html`
- Short presentation for exam discussion

### Key Tasks

#### 1) Data Loading & Cleaning
- Import dataset and display first few rows
- Identify variable names and data types (numeric, categorical, etc.)
- Count observations and variables
- Detect missing values and their proportions
- Decide on missing value handling strategy (removal, imputation, etc.)

#### 2) Univariate Data Description
- Compute descriptive statistics (mean, median, min, max, standard deviation)
- Create frequency tables for categorical variables
- Plot histograms and boxplots for numeric variables
- Analyze distribution shapes
- Create bar/pie charts for categorical variables
- Identify most frequent categories

#### 3) Bivariate and Multivariate Description
- Compute correlations among numeric variables
- Visualize correlations with heatmaps
- Create scatterplots for numeric variable pairs
- Identify linear and nonlinear relationships
- Compare numeric distributions across categories
- Build contingency tables for multiple categorical variables

#### 4) Principal Component Analysis and Clustering
- Scale and encode variables
- Apply Principal Component Analysis (PCA)
- Determine components needed for ≥80% variance explanation
- Visualize loading vectors on correlation circle
- Create biplots and analyze patterns
- Select and apply clustering algorithm
- Determine optimal number of clusters
- Visualize clusters and assess separation quality

#### 5) Summary and Interpretation
- Summarize main dataset characteristics
- Identify interesting patterns and differences
- Provide interpretative commentary for each section

---

## Assignment 2: Probability Models Comparison

### Objective
Go in-depth on a selected pair of probability distributions, exploring their relationships, differences, and real-world applications.

### Available Model Pairs
- Exponential and Poisson
- Binomial and Pascal
- Beta and Gamma
- Weibull and Gamma

### Deliverables
- Analysis file (`.ipynb` or `.Rmd`)
- Export formats: `.pdf` or `.html`
- Short presentation for exam discussion

### Key Tasks

#### 1) Theoretical Comparison
- Write probability density/mass functions for each distribution
- Identify and describe all parameters
- Report mean and variance expressions
- Explore parameter dependency on mean and variance
- Plot distributions for various parameter values
- Analyze effects of parameter changes (symmetry, spread, skewness)
- Explain practical meaning of parameters in real-world contexts
- Discuss connections to Gaussian (Normal) distribution
- Analyze convergence and asymptotic behavior

#### 2) Simulation Study
- Generate random samples (e.g., 1000 observations) from each distribution
- Test multiple parameter combinations
- Plot histograms and density plots
- Compare simulated data to theoretical distributions
- Compute sample mean and variance
- Compare empirical results with theoretical values
- Discuss consistency with theory

#### 3) Application Scenario
- Develop realistic or hypothetical use case
- Describe the nature of random variables
- Justify distribution assumptions
- Explain parameter interpretation
- Simulate and analyze a dataset consistent with scenario

### Suggested Tools
**R:**
- `rbeta()`, `rgamma()`, `rbinom()`, `rpois()`, `rexp()`
- Use `?` function for help

**Python:**
- `scipy.stats`: `beta`, `gamma`, `binom`, `poisson`, `expon`
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)

**LaTeX:**
- [LaTeX in 30 minutes](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)

---

## Assignment 3: Regression and Data Simulation

### Objective
Simulate datasets and fit various regression models, perform diagnostics, and provide statistical interpretations across three thematic blocks.

### Deliverables
- Analysis file (`.ipynb` or `.Rmd`)
- Export formats: `.pdf` or `.html`
- Short presentation for exam discussion

### Key Tasks

#### 1) Dataset Simulation and Linear Regression
- Simulate data from linear function: p=10 predictors, n=300 observations
- Assign meaningful effects to subset of variables (larger coefficients to important variables, smaller/null to others)
- Document chosen function with justifications
- Fit Linear Regression model
- Report coefficients, standard errors, p-values, t-statistics
- Identify statistically significant predictors
- Validate match between significant variables and originally set strong coefficients
- Perform regression diagnostics using residuals
- Identify high leverage points, influential observations, and outliers
- Comment on model stability impact

#### 2) Shrinkage and Regularization
- Fit Ridge Regression model with cross-validation for optimal lambda
- Plot coefficient paths
- Compare estimated vs. true generating coefficients
- Fit Lasso Regression model with cross-validation for optimal lambda
- Identify coefficients that shrink exactly to zero
- Compare variable selection with true generating beta
- Fit Elastic-Net Regression model
- Use grid-search or graphical approach for optimal alpha and lambda combination

#### 3) Non-Linear Regression and Bias-Variance Experiment
- Select non-linear generating function (exponential, polynomial, trigonometric, or combination)
- Simulate dataset: p=1 predictor, n=10 observations
- Plot true function and observed points
- Fit polynomial regression models: one with degree < 10, one with degree = 10
- Compare regression curves, true function, and observed points
- Identify model with lowest train error
- Identify model closest to true generating function
- Explain why degree-10 polynomial interpolates all points
- Describe and comment on bias-variance trade-off
- Simulate new dataset with p=1, n>100 using same underlying function
- Fit degree-10 polynomial regression again
- Compare overfitting between small and large sample models
- Fit non-linear regression models from course lecture
- Find optimal parameters and compare performances

---

## Assignment 4: Classification & Final Project

### Objective
Develop a complete data analysis project on a classification problem of choice, showcasing learned skills and analytical reasoning.

### Requirements
- Supervised classification task
- Dataset of choice (public or provided datasets accepted)
- Clear narrative explaining approach and reasoning
- Well-motivated decisions on included/excluded sections

### Deliverables
- Analysis file (`.ipynb` or `.Rmd`)
- Export formats: `.pdf` or `.html`
- Short presentation for exam discussion

### Suggested Structure

#### Real-World Motivation & Problem Definition
- Describe real-world context
- Explain why the problem matters
- Define response variable and classes
- Specify classification setting type
- Justify dataset choice

#### Exploratory Data Analysis
- Inspect dataset structure and variables
- Generate summary statistics
- Create exploratory plots
- Highlight potential challenges

#### Preprocessing Pipeline
- Split data into train/test sets
- Handle missing values
- Encode categorical variables
- Scale features
- Engineer new features
- Address class imbalance
- Apply dimensionality reduction if needed

#### Unsupervised Analysis
- Perform clustering to explore structure
- Apply dimensionality reduction for visualization
- Integrate clustering into preprocessing or feature engineering

#### Model Building
- Select and justify chosen models
- Explain intuition behind each model
- Specify hyperparameters for tuning

#### Model Tuning and Resampling
- Perform k-fold cross-validation
- Execute grid search for hyperparameter optimization
- Compare validation curves

#### Model Evaluation
- Select appropriate evaluation metrics
- Present results for single or multiple models
- Explain interpretation of results

#### Interpretation and Insights
- Analyze variable importance
- Visualize decision boundaries
- Create tree visualizations (if applicable)
- Extract actionable insights

#### Final Discussion
- Summarize findings
- Discuss insights about the data
- Evaluate what worked and what didn't
- Assess model performance adequacy
- Suggest improvements with more time/data
- Discuss real-world implications

---

## Team Composition Requirements

All assignments support team work (up to 3 members) with requirements for heterogeneous composition:
- Diverse bachelor degree backgrounds
- Diverse country of origin

---

## General Guidelines

### Output Quality
- All plots must have clear titles, labels, and legends (where applicable)
- All equations should be well-formatted
- Each section should include interpretative commentary
- Presentations should focus on interpretation and communication, not code reproduction
- Use clean visuals and minimal text in presentations

### Tools and Technologies
Projects can be completed in:
- **Python** with Jupyter Notebooks (`.ipynb`)
- **R** with R Markdown (`.Rmd`)

### Submission Format
- Primary: `.ipynb` or `.Rmd`
- Export: `.pdf` or `.html`
- Plus: presentation file for exam discussion

---

## Submission Links

- [Assignment 1 Submission](https://forms.gle/8qfUfk9FcDfUcDdw8)
- [Assignment 2 Submission](https://forms.gle/ZRVqqdeo8RkeY2oy8)
- [Assignment 3 Submission](https://forms.gle/JT7EzCohvnq58wPDA)
- [Assignment 4 Submission](https://forms.gle/RJ8gGh6mkzikxG2c8)

---

## Course Topics Covered

- **Exploratory Data Analysis**: Descriptive statistics, visualization, correlation analysis
- **Probability Models**: Distribution theory, simulation, parameter estimation
- **Regression Analysis**: Linear, Ridge, Lasso, Elastic-Net, polynomial, non-linear regression
- **Diagnostics & Validation**: Residual analysis, cross-validation, model assessment
- **Classification**: Supervised learning, model selection, performance metrics
- **Dimensionality Reduction**: Principal Component Analysis (PCA), clustering
- **Data Preprocessing**: Missing value handling, feature engineering, scaling, encoding

---

## Project Structure
