# Feature Selection Demo 🔍

A comprehensive demonstration of feature selection techniques for high-dimensional machine learning datasets using scikit-learn.

## 📋 Overview

This project demonstrates three popular feature selection methods applied to synthetic high-dimensional data:

1. **Variance Threshold**: Removes features with low variance
2. **SelectKBest (Univariate Selection)**: Uses ANOVA F-test to select top features
3. **Recursive Feature Elimination (RFE)**: Wrapper method using logistic regression

The demo includes both an interactive Jupyter notebook with visualizations and a standalone Python script.

## 🚀 Features

- **Interactive Analysis**: Step-by-step feature selection pipeline in Jupyter notebook
- **Data Visualization**: Variance analysis and scatter plots for selected features
- **Multiple Techniques**: Comparison of filter, univariate, and wrapper methods
- **Synthetic Dataset**: 200 samples, 100 features (only 10 informative)
- **Educational Content**: Clear explanations and visualizations for learning

## 📁 Project Structure

```
feature-selection-demo/
├── feature_selection_demo.ipynb    # Interactive Jupyter notebook with visualizations
├── feature_selection_demo.py       # Standalone Python script
├── requirements.txt                # Project dependencies
├── synthetic_features.csv          # Generated synthetic feature data (200×100)
├── synthetic_target.csv           # Generated binary target labels
└── README.md                       # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/feature-selection-demo.git
   cd feature-selection-demo
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Jupyter notebook (additional dependencies):**
   ```bash
   pip install jupyter matplotlib seaborn pandas
   ```

## 📊 Usage

### Option 1: Interactive Jupyter Notebook (Recommended)

```bash
jupyter notebook feature_selection_demo.ipynb
```

The notebook provides:
- Step-by-step feature selection workflow
- Data generation and preprocessing
- Variance analysis with bar charts
- Scatter plots of selected features
- Detailed explanations of each method

### Option 2: Python Script

```bash
python feature_selection_demo.py
```

## 🧪 Feature Selection Pipeline

### Step 1: Data Generation
```python
# Creates synthetic dataset: 200 samples × 100 features (10 informative)
X, y = make_classification(n_samples=200, n_features=100, n_informative=10, random_state=0)
```

### Step 2: Variance Threshold Filtering
```python
# Removes features with variance < 0.1
sel_var = VarianceThreshold(threshold=0.1)
X_var = sel_var.fit_transform(X)
# Result: 200 × ~85 features
```

### Step 3: Univariate Feature Selection
```python
# Selects top 20 features using ANOVA F-test
sel_k = SelectKBest(score_func=f_classif, k=20)
X_k = sel_k.fit_transform(X_var, y)
# Result: 200 × 20 features
```

### Step 4: Recursive Feature Elimination
```python
# Further reduces to 5 best features using logistic regression
lr = LogisticRegression(max_iter=500, solver='liblinear')
sel_rfe = RFE(lr, n_features_to_select=5, step=0.1)
sel_rfe.fit(X_k, y)
# Result: 200 × 5 features
```

## 📈 Sample Results

```
Original shape: (200, 100)
After VarianceThreshold: (200, 85)
After SelectKBest (k=20): (200, 20)
RFE selected features: 5 most predictive features
```

## 🎨 Visualizations

The notebook includes several informative plots:

1. **Variance Bar Chart**: Shows variance distribution across the first 30 features
   - Red dashed line indicates the variance threshold (0.1)
   - Helps identify low-variance features to remove

2. **Feature Scatter Plots**: Displays the top 5 selected features
   - Points colored by target class (binary classification)
   - Shows how well selected features separate the classes

## 🔧 Methods Explained

### Variance Threshold (Filter Method)
- **Purpose**: Remove features with low variance (likely uninformative)
- **Advantages**: Fast, model-agnostic
- **Use case**: Initial filtering step

### SelectKBest (Univariate Selection)
- **Purpose**: Select features based on univariate statistical tests
- **Method**: ANOVA F-test for classification
- **Advantages**: Computationally efficient, good baseline
- **Use case**: Reducing dimensionality while preserving predictive features

### Recursive Feature Elimination (Wrapper Method)
- **Purpose**: Select features based on model performance
- **Method**: Iteratively removes least important features
- **Advantages**: Considers feature interactions
- **Use case**: Fine-tuning feature set for specific model

## 📚 Dependencies

### Core Dependencies (requirements.txt)
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning algorithms and feature selection

### Additional for Notebook
- `pandas`: Data manipulation and analysis
- `matplotlib`: Basic plotting functionality
- `seaborn`: Statistical data visualization
- `jupyter`: Interactive notebook environment

## 🎯 Learning Outcomes

After running this demo, you'll understand:
- When and how to apply different feature selection techniques
- The trade-offs between filter, wrapper, and embedded methods
- How to visualize and interpret feature selection results
- The impact of feature selection on high-dimensional datasets
- Best practices for feature selection pipelines

## 🔍 Use Cases

This demo is perfect for:
- **Students** learning machine learning and feature engineering
- **Data Scientists** exploring feature selection techniques
- **Researchers** comparing different selection methods
- **Practitioners** building feature selection pipelines
- **Educators** teaching dimensionality reduction concepts

## 🤝 Contributing

Contributions are welcome! Ideas for enhancement:
- Add embedded methods (L1 regularization, tree-based importance)
- Include real-world datasets
- Add cross-validation for feature selection
- Implement mutual information-based selection
- Add performance comparison metrics

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Resources

- [Scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Feature Engineering and Selection: A Practical Approach](http://www.feat.engineering/)
- [Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

---

⭐ **If you found this helpful, please star the repository!**

📧 **Questions or suggestions?** Feel free to open an issue or reach out!
