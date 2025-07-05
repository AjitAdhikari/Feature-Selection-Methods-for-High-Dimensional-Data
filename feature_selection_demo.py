#!/usr/bin/env python3
"""
Simple demo: Feature selection on high-dimensional synthetic data
Methods: VarianceThreshold, SelectKBest, RFE
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

def main():
    # Generate synthetic high-dimensional dataset
    X, y = make_classification(n_samples=200, n_features=100, n_informative=10, random_state=0)

    print(f"Original shape: {X.shape}")

    # 1. Filter by variance
    sel_var = VarianceThreshold(threshold=0.1)
    X_var = sel_var.fit_transform(X)
    print(f"After VarianceThreshold: {X_var.shape}")

    # 2. Filter by univariate ANOVA F-test
    sel_k = SelectKBest(score_func=f_classif, k=20)
    X_k = sel_k.fit_transform(X_var, y)
    print(f"After SelectKBest (k=20): {X_k.shape}")
    print("Top 20 feature indices:", sel_k.get_support(indices=True))

    # 3. Wrapper RFE with logistic regression
    lr = LogisticRegression(max_iter=500, solver='liblinear')
    sel_rfe = RFE(lr, n_features_to_select=5, step=0.1)
    sel_rfe.fit(X_k, y)
    print("RFE selected indices (relative to SelectKBest subset):", sel_rfe.get_support(indices=True))

if __name__ == "__main__":
    main()
