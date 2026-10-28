| Model | Threshold Policy | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TP / FP / FN / TN | Data Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `logistic_regression` | Default (0.50) | 0.50 | 0.7500 | 0.4615 | 0.9231 | 0.6154 | 0.8457 | 24 / 28 / 2 / 66 | `synthetic_fixture` |
| `logistic_regression` | Val-Tuned | 0.38 | 0.6667 | 0.3906 | 0.9615 | 0.5556 | 0.8457 | 25 / 39 / 1 / 55 | `synthetic_fixture` |
| `random_forest` | Default (0.50) | 0.50 | 0.8417 | 0.6522 | 0.5769 | 0.6122 | 0.8629 | 15 / 8 / 11 / 86 | `synthetic_fixture` |
| `random_forest` | Val-Tuned | 0.38 | 0.7417 | 0.4444 | 0.7692 | 0.5634 | 0.8629 | 20 / 25 / 6 / 69 | `synthetic_fixture` |
