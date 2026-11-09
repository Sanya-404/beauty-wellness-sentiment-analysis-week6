"""
Beauty & Wellness Customer Sentiment & Feedback Analysis
Week 6: Model Evaluation, Executive Reporting & Business ROI Calculation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate_business_impact(y_true, y_pred, y_prob):
    """
    Translates standard machine learning evaluation metrics into
    non-technical executive business impact indicators:
    - Critical Defect Catch Rate (Recall)
    - False Alarm vs Alarm Reliability (Precision)
    - Financial ROI & Product Return Savings
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    total = len(y_true)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    auc = roc_auc_score(y_true, y_prob)
    
    print("=" * 70)
    print("       WEEK 6: EXECUTIVE MODEL EVALUATION & BUSINESS IMPACT")
    print("=" * 70)
    print(f"Total Customer Reviews Processed:       {total:,}")
    print(f"Defect / Hazard Catch Rate (Recall):    {recall * 100:.2f}%")
    print(f"Alert Reliability (Precision):          {precision * 100:.2f}%")
    print(f"Overall Model Discriminating Power:     {auc * 100:.2f}%")
    print("-" * 70)
    print("CONFUSION MATRIX BREAKDOWN (EXECUTIVE VIEW):")
    print(f" - True Positives (Defects Correctly Caught):      {tp:,}")
    print(f" - False Positives (False Alarms):                 {fp:,}")
    print(f" - False Negatives (Missed Defects / Slippage):    {fn:,}")
    print(f" - True Negatives (Satisfied Reviews Verified):    {tn:,}")
    print("-" * 70)
    
    # Financial Impact & Brand ROI Model
    # Assumes average return cost of $25 per returned cosmetic item
    # Model intervention achieves 40% customer retention recovery
    avg_return_cost = 25.0
    recovery_rate = 0.40
    estimated_cost_savings = tp * avg_return_cost * recovery_rate
    
    print("COMMERCIAL ROI & COST MITIGATION ANALYSIS:")
    print(f" - Estimated Return Processing Cost per Unit:     ${avg_return_cost:.2f}")
    print(f" - Proactive Outreach Recovery Rate:             {recovery_rate * 100:.0f}%")
    print(f" - Total Projected Annual Cost Savings:          ${estimated_cost_savings:,.2f}")
    print("=" * 70)
    
    return {
        "recall": recall,
        "precision": precision,
        "roc_auc": auc,
        "projected_savings_usd": estimated_cost_savings
    }

def run_simulation():
    # Representative holdout test partition (2,000 Defect reviews, 8,000 Satisfied reviews)
    np.random.seed(42)
    n_samples = 10000
    # 20% negative / defect class, 80% positive class
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.20, 0.80])
    
    # Simulate high-performing logistic regression probabilities with class weighting
    y_prob = np.zeros(n_samples)
    y_prob[y_true == 1] = np.random.beta(a=7, b=2, size=np.sum(y_true == 1))
    y_prob[y_true == 0] = np.random.beta(a=2, b=7, size=np.sum(y_true == 0))
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    evaluate_business_impact(y_true, y_pred, y_prob)

if __name__ == "__main__":
    run_simulation()
