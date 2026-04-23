## AutoML Framework Comparison (10 Systems)

Protocol: 2-fold × 2 seeds = 4 evaluations per (system, dataset).  
Datasets: iris (150×4), wine (178×13), breast\_cancer (569×30), digits (1797×64).

> **Note:** TPOT and FLAML excluded due to platform constraints  
> (TPOT requires PyTorch which exceeds available virtual memory;  
> FLAML ignores `time_budget` on Windows/Python 3.13, taking >298 s  
> for a 10 s budget). H2O excluded due to insufficient disk space  
> (requires ~500 MB; C: has <256 MB free).

### Mean Accuracy

| System | breast_cancer | digits | iris | wine | Mean |
| --- | --- | --- | --- | --- | --- |
| OptunaEnsemble | 0.9772 | 0.9858 | 0.9400 | 0.9775 | 0.9701 |
| GreedyEnsemble | 0.9754 | 0.9866 | 0.9367 | 0.9747 | 0.9684 |
| AutoStack | 0.9798 | 0.9777 | 0.9367 | 0.9747 | 0.9672 |
| FeatEngAutoML | 0.9789 | 0.9850 | 0.9333 | 0.9719 | 0.9673 |
| **C60.ai** | **0.9499** | **0.9883** | **0.9600** | **0.9579** | **0.9640** |
| BayesSearchCV | 0.9701 | 0.9819 | 0.9433 | 0.9663 | 0.9654 |
| OptunaSearch | 0.9570 | 0.9761 | 0.9600 | 0.9719 | 0.9662 |
| BroadRandomSearch | 0.9692 | 0.9752 | 0.9600 | 0.9579 | 0.9656 |
| HyperoptSearch | 0.9604 | 0.9769 | 0.9533 | 0.9635 | 0.9635 |
| HalvingGrid | 0.9710 | 0.9789 | 0.9400 | 0.9466 | 0.9591 |
| SuccessiveHalving | 0.9508 | 0.9741 | 0.9567 | 0.9663 | 0.9620 |

### Average Rank (lower = better)

| Rank | System | Avg Rank |
| --- | --- | --- |
| 1 | OptunaEnsemble | 3.50 |
| 2 | GreedyEnsemble | 4.50 |
| 3 | AutoStack | 5.00 |
| 4 | FeatEngAutoML | 5.25 |
| 5 | C60.ai ** | 6.00 |
| 6 | BayesSearchCV | 6.00 |
| 7 | OptunaSearch | 6.25 |
| 8 | BroadRandomSearch | 7.00 |
| 9 | HyperoptSearch | 7.25 |
| 10 | HalvingGrid | 7.50 |
| 11 | SuccessiveHalving | 7.75 |

### Statistical Significance (Wilcoxon Signed-Rank, C60.ai vs each system)

H₁: C60.ai accuracy > baseline, one-sided, α = 0.05.

| Baseline | Mean diff | p-value | Significant? |
| --- | --- | --- | --- |
| HalvingGrid | +0.0049 | 0.2507 | No |
| SuccessiveHalving | +0.0021 | 0.4181 | No |
| HyperoptSearch | +0.0005 | 0.4691 | No |
| BroadRandomSearch | -0.0016 | 0.4897 | No |
| OptunaSearch | -0.0022 | 0.5699 | No |
| BayesSearchCV | -0.0014 | 0.5920 | No |
| FeatEngAutoML | -0.0032 | 0.6657 | No |
| GreedyEnsemble | -0.0043 | 0.7681 | No |
| AutoStack | -0.0032 | 0.7887 | No |
| OptunaEnsemble | -0.0061 | 0.8033 | No |