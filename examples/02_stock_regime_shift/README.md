# Example 2: Stock Market Regime Change Detection (Time Series → Classification)

**Use Case Type**: Financial time-series classification

**Input Format**: OHLCV + macroeconomic signals

**Pipeline Structure**:
- DAG with temporal window slicing + feature extractors (volatility, MACD, RSI)
- Transformer-based classifier node via `node_nas.py`
- RL-optimized final layer for Sharpe-adjusted accuracy

**Advanced Features Triggered**:
- `meta_dag_agent.py` (market shift adaptation)
- `quantum_dag_opt.py` (hyperparameter tuning)
- `online_adapter.py` (streaming updates)

**Expected Output**:
- Change regime detector (bear/bull/sideways)
- Online-updating pipeline
- Evaluation notebook with rolling backtest

---

## Usage

- Pipeline YAML: [`pipeline_dag.yaml`](./pipeline_dag.yaml)
- Walkthrough Notebook: [`notebook.ipynb`](./notebook.ipynb)

**Run via CLI:**

```bash
python c60_executor.py --input examples/02_stock_regime_shift/pipeline_dag.yaml
```

**Or in Python:**

```python
from c60.engine.agent import AutoDAGAgent
agent = AutoDAGAgent()
agent.load_yaml('examples/02_stock_regime_shift/pipeline_dag.yaml')
agent.run()
```
