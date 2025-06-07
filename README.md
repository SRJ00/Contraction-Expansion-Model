# Contraction-Expansion Model

This repository extends the original **m-neighbor classification with contamination-based rejection** by introducing a boundary **expansion mechanism**.

### 🔄 What's New
While the original paper focuses purely on *contraction* of class boundaries based on contamination rates, this work adds support for **expansion**. This enables more flexible classification by allowing the model to include slightly anomalous but valid observations, which improves robustness in certain data distributions.

We achieve this by modifying the class-specific threshold computation to optionally:
- Contract using a lower percentile of deviation scores (as in the original work), or
- Expand the boundary using the maximum deviation score plus a small quantile margin.

### 📎 Reference to Original Paper
> **m-Neighbor Classification with Contamination-Based Rejection**  
> [Original paper link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5267814)

### 📁 Contents
- `contraction_expansion_model.py` — Python implementation of the extended model
- `Contraction_Expansion_Model.pdf` — PDF with theoretical explanation and algorithm in LaTeX
- `LICENSE` — Open-source license

### 📜 License
This work is open-sourced under the MIT License.
