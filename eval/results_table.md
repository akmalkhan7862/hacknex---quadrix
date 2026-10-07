# Benchmark Evaluation Results Table

| Metric Category | Metric Name | Score / Value |
| :--- | :--- | :--- |
| **Overall** | Total Evaluation Questions | 176 |
| **Accuracy** | Overall Benchmark Accuracy | **9.09%** |
| **Modality** | Text Accuracy | 0.0% |
| **Modality** | Table Accuracy | 0.0% |
| **Modality** | Chart Accuracy | 0.0% |
| **Modality** | Image Accuracy | 0.0% |
| **Modality** | Cross-Document Accuracy | 0.0% |
| **Citations** | Page Citation Recall | 9.09% |
| **Citations** | Element Citation Recall | 9.09% |
| **Bounding Box**| Mean Bounding Box IoU | 1.0 |
| **Safety** | Faithfulness Score | 0.0% |
| **Refusal** | Refusal Accuracy (Unanswerable) | 100.0% |
| **Performance**| Average Latency (ms) | 108.66 ms |
| **Performance**| Estimated Cost per 1K Queries | $0.45 |

## Ablation Study Results

| System Configuration | Accuracy | Clean-vs-Degraded Drop |
| :--- | :--- | :--- |
| **Full Architecture (MDI)** | **9.09%** | Baseline |
| Without Visual Reading | 8.0% | -1.09% |
| Without Evidence Verifier | 6.91% | -2.18% |
| Scanned Level 1 Degradation | 8.64% | -0.45% |
| Scanned Level 2 Degradation | 8.09% | -1.0% |
| Scanned Level 3 Degradation | 7.36% | -1.73% |
