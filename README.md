# Contraction-Expansion Model

This repository extends the original **m-neighbor classification with contamination-based rejection** by introducing a boundary **expansion mechanism**.

### 🔄 What's New
While the original paper focuses purely on *contraction* of class boundaries based on contamination rates, this work adds support for **expansion**. This enables more flexible classification by allowing the model to include slightly anomalous but valid observations, which improves robustness in certain data distributions.

We achieve this by modifying the class-specific threshold computation to optionally:
- Contract using a lower percentile of deviation scores (as in the original work), or
- Expand the boundary using the maximum deviation score plus a small quantile margin.

### 🧪 Example Usage
```python
from contraction_expansion_model import outlier_contraction_expansion_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create synthetic data
X, y = make_classification(n_samples=500, n_features=5, n_informative=4, 
                           n_redundant=0, n_classes=3, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize model
model = outlier_contraction_expansion_classifier(
    m_neighbor=3,
    lambda_weight=0.5,
    threshold_type='expansion',  # or 'contraction'
    rate=0.1
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)



# Evaluate performance
accuracy = model.score(y_test, y_pred, accuracy_metric='accuracy')
print(f"\nOverall Accuracy (including rejection): {accuracy:.2f}")
```

### 📎 Reference to Original Paper
> **m-Neighbor Classification with Contamination-Based Rejection**  
> [Original paper link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5267814)

### 📁 Contents
- `contraction_expansion_model.py` — Python implementation of the extended model
- `Contraction_Expansion_Model.pdf` — PDF with theoretical explanation and algorithm in LaTeX
- `LICENSE` — Open-source license

### 📜 License
This work is open-sourced under the MIT License.
