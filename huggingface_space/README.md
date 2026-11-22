---
title: Enterprise AI Support Ticket Classifier
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# 📧 Enterprise Email / Support Ticket Classification System

An enterprise NLP machine learning application for multi-class support ticket categorization into **6 operational business categories**:
1. **Billing & Finance**
2. **Customer Service & Returns**
3. **Human Resources**
4. **Product Support**
5. **Sales & General**
6. **Technical Support**

## 🔬 Scientific Highlights
- **100% Zero Data Leakage**: Strictly deduplicated across 59,577 samples.
- **Model**: Calibrated Max-Margin Linear Support Vector Classifier (`CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced'))`).
- **Features**: Sublinear TF-IDF (1, 2) N-Grams (75,000 features).
- **Macro Precision**: 79.04%
- **Macro F1-Score**: 72.21%
- **Test Accuracy**: 72.00%
