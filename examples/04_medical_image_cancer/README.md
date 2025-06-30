# Example 4: Medical Image Cancer Classifier (Image Input)

**Use Case Type**: Image classification (histopathology)

**Input Format**: .PNG / .TIFF images of tumor slides

**Pipeline Structure**:
- DAG contains CNN extractors (ResNet → EfficientNet)
- NAS-driven architecture tuner on CNN layer depth
- Final SVM or dense layer based on features

**Advanced Features Triggered**:
- `node_nas.py` (CNN NAS tuning)
- `audit.py` (diagnostic trace)
- `ethics_check.py` (bias/fairness)

**Expected Output**:
- Trained cancer classifier
- Visual diagnostic overlay (Grad-CAM)
- Bias/fairness report

---

## Usage

- Pipeline YAML: [`pipeline_dag.yaml`](./pipeline_dag.yaml)
- Walkthrough Notebook: [`notebook.ipynb`](./notebook.ipynb)

**Run via CLI:**

```bash
python c60_executor.py --input examples/04_medical_image_cancer/pipeline_dag.yaml
```

**Or in Python:**

```python
from c60.engine.agent import AutoDAGAgent
agent = AutoDAGAgent()
agent.load_yaml('examples/04_medical_image_cancer/pipeline_dag.yaml')
agent.run()
```
