# UCDI: Uncertainty-Calibrated Decision Intelligence

Code and data for the short communication
**"Selective Decision Intelligence Under Predictive Uncertainty: Calibration, Human Deferral, and Error Concentration in AI Automation"**
(Srihitha Reddy Betelly, submitted to *Knowledge-Based Systems*).

UCDI decides which credit decisions an AI model should automate and which it should defer to a human reviewer. It ranks cases by the uncertainty of a calibrated probability, `u(x) = 1 - |2p(x) - 1|`, then evaluates the trade-off between automation coverage, error on the automated cases, and how many of the model's errors end up in the deferred cases.

## Reproduce

```bash
pip install -r requirements.txt
python ucdi.py            # about 2 minutes on a laptop CPU
```

Options: `--coverage 0.80` (automation share), `--seeds 10` (robustness splits), `--trials 1000` (random-deferral permutations), `--seed 42`.

## Repository layout

| Path | Contents |
|---|---|
| `ucdi.py` | Full pipeline: data split, models, calibration, selective prediction, permutation test, seeds, figures |
| `data/credit_default_taiwan.csv` | UCI "Default of Credit Card Clients" dataset (30,000 x 23 + target), converted from the original `.xls` |
| `results/table1_models.csv` | Table 1: discrimination, calibration and selective metrics for 9 model variants |
| `results/table2_selective.csv` | Table 2: risk/coverage and error capture vs random deferral |
| `results/seeds.csv`, `results/summary.json` | 10-seed robustness and run summary |
| `figures/` | Fig. 1 reliability diagrams, Fig. 2 risk-coverage / error concentration, graphical abstract |

## Protocol

* Stratified, non-overlapping 60/20/20 split: train (18,000), calibration (6,000), test (6,000).
* Models: logistic regression, random forest, histogram gradient boosting (scikit-learn), each raw, sigmoid (Platt) and isotonic calibrated on the calibration split.
* The decision model is picked on the **calibration split** (lowest Brier score of the raw models), so the test set is never used for selection.
* Random-deferral baseline: 1,000 random subsets of the same size; empirical p-value = (1 + #random >= UCDI) / (1 + trials).

## Data

Yeh, I.-C. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Licensed CC BY 4.0.

## License

Code: MIT. Data: CC BY 4.0 (original authors).
