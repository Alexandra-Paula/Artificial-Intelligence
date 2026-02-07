# Copilot Instructions for AI Lab 1 - E-commerce Customer Analysis

## Project Overview
This is a machine learning laboratory assignment analyzing e-commerce customer data to predict yearly spending using linear regression. The project uses a single Jupyter notebook workflow in `Lab1/Lab1.ipynb` with accompanying customer data in `Lab1/Ecommerce Customers.txt`.

## Data Pipeline & Architecture

### Data Source: `Ecommerce Customers.txt`
- CSV format with 1002 customer records
- Key features:
  - `Avg. Session Length`: Average session duration in minutes
  - `Time on App`: Minutes spent on mobile app
  - `Time on Website`: Minutes spent on website
  - `Length of Membership`: Years as a customer
  - `Yearly Amount Spent`: Target variable (what we predict)
- Additional metadata: Email, Address, Avatar (not used for modeling)

### Workflow Pattern in Lab1.ipynb
The notebook follows a strict ML pipeline structure:
1. **Data Loading & Exploration** (cells 1-9): Load CSV, display head/info/describe, exploratory plots
2. **Relationship Analysis** (cells 6-11): Use `sns.jointplot()` and `sns.pairplot()` to identify feature correlations
3. **Data Splitting** (cells 13-14): Create train/test sets with 70/30 split, fixed random_state=101
4. **Model Training** (cells 16-18): Fit LinearRegression, extract coefficients
5. **Prediction & Evaluation** (cells 20-22): Generate predictions, visualize scatter plot, calculate residuals

## Critical Development Patterns

### Variable Naming Issues to Watch
- There's a bug in cell 3: code references `data.info()` but dataframe is named `customers`
- Cell 20 uses variable `x_test` for predictions (lowercase), while input is `X_test` (uppercase)
- When fixing, maintain naming consistency: use `X_train`, `X_test`, `y_train`, `y_test` throughout

### Pandas-to-DataFrame Conversions
- Coefficients are created as DataFrame via `pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])`
- This pattern converts 1D array to labeled column, useful for interpretability
- When extending: follow same pattern for residuals or other metrics

### Visualization Conventions
- Exploratory plots: use `sns.jointplot()` and `sns.pairplot()` with default settings
- Predictive plots: scatter plots with `plt.scatter(y_test, predictions)` for residual inspection
- Always call `plt.show()` after plotting to ensure output appears in notebook

## Common Tasks & Commands

### Running the Notebook
- Execute cells sequentially - later cells depend on earlier variable definitions
- Cell 1 must run first (imports and data loading)
- If kernel needs restart: Variables list includes LinearRegression, X, X_test, y, y_test, etc.

### Adding Analysis or Metrics
When adding model evaluation (Mean Squared Error, R² score, residual analysis):
- Use sklearn metrics: `from sklearn.metrics import mean_squared_error, r2_score`
- Calculate on test set: `r2_score(y_test, predictions)`
- Append as new code cells after cell 21 (prediction visualization)

### Modifying Features or Model
- Feature selection: Change X definition in cell 13 (add/remove columns)
- Model swap: Replace `LinearRegression` in cell 16 with other sklearn estimators (Lasso, Ridge)
- Train/test split: Adjust `test_size` parameter in cell 14 (currently 0.3)

## Dependencies
- pandas: Data loading and manipulation
- numpy: Numerical operations (already imported via pandas)
- matplotlib.pyplot: Basic plotting
- seaborn (sns): Statistical visualization (jointplot, pairplot, lmplot)
- scikit-learn: Model training, train_test_split, LinearRegression

## Notes for Code Generation
- Preserve markdown cells for section organization and learning notes (Romanian language)
- New analysis code should follow the same sklearn/seaborn patterns already established
- Keep variable names aligned with existing notebook (X/y convention for features/target)
- Test any new cells with actual data before suggesting - kernel variables show all available objects
