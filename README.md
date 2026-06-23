# TabFM: Tabular Foundation Models

TabFM (Tabular Foundation Model) is a scikit-learn compatible tabular foundation model. It allows you to perform zero-shot classification and regression on tabular datasets with mixed column types out-of-the-box.

At inference time, TabFM does not require training parameters on your dataset; instead, it leverages in-context learning by reading your training data as "context" to make instant predictions on new test samples.

*This is not an officially supported Google product.*

---

## Installation

To install TabFM, clone the repository and install it locally:

```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .
```

The base install uses CPU-only JAX. For GPU execution, install the `cuda`
extra to pull the CUDA 12 plugin and NVIDIA libraries:

```bash
pip install -e .[cuda]
```

### Requirements
For a complete list of pinned dependencies and versions, please see [requirements.txt](requirements.txt). The core requirements are:
*   Python >= 3.11
*   JAX (specifically `jax==0.10.1`)
*   Flax (specifically `flax==0.12.7`, using the modern `flax.nnx` API)
*   Hugging Face Hub (for downloading pre-trained weights)

---

## Quick Start (TabFM v1.0.0)

We provide pre-trained weights for the **TabFM v1.0.0** release. The library handles downloading and loading these weights automatically using the `tabfm_v1_0_0` model release package.

### 1. Classification Example

```python
import numpy as np
import pandas as pd
from tabfm import tabfm_v1_0_0
from tabfm import TabFMClassifier

# 1. Load pre-trained TabFM v1.0.0 model (automatically downloads from Hugging Face)
model = tabfm_v1_0_0.load()

# 2. Initialize scikit-learn compatible classifier
clf = TabFMClassifier(model=model)

# 3. Prepare your dataset (supports mixed numerical and categorical features)
X_train = pd.DataFrame({
    "age": [25.0, 45.0, 35.0, 50.0],
    "job": ["engineer", "manager", "engineer", "manager"],
    "income": [80000, 120000, 90000, 130000]
})
y_train = np.array(["low_risk", "high_risk", "low_risk", "high_risk"])

X_test = pd.DataFrame({
    "age": [30.0, 48.0],
    "job": ["engineer", "manager"],
    "income": [85000, 125000]
})

# 4. Fit classifier (prepares ordinal encoders and numerical scalers)
clf.fit(X_train, y_train)

# 5. Predict classes and probabilities
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

print("Predictions:", predictions)
print("Class Probabilities:\n", probabilities)
```

### 2. Regression Example

```python
import numpy as np
import pandas as pd
from tabfm import tabfm_v1_0_0
from tabfm import TabFMRegressor

# 1. Load pre-trained TabFM v1.0.0 model
model = tabfm_v1_0_0.load()

# 2. Initialize scikit-learn compatible regressor
reg = TabFMRegressor(model=model)

# 3. Prepare your dataset
X_train = pd.DataFrame({
    "sqft": [1200, 2500, 1500, 3000],
    "neighborhood": ["A", "B", "A", "C"]
})
y_train = np.array([250000, 550000, 310000, 620000])

X_test = pd.DataFrame({
    "sqft": [1800, 2800],
    "neighborhood": ["A", "B"]
})

# 4. Fit and Predict
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)

print("Predicted Prices:", predictions)
```

---

## Examples Directory

You can find runnable scripts for both classification and regression under the [examples/](examples/) folder:
*   [classification_example.py](examples/classification_example.py)
*   [regression_example.py](examples/regression_example.py)

To run them, simply execute:
```bash
python examples/classification_example.py
```

---

## Running Tests

To run the unit tests using Bazel:
```bash
bazel test //...
```
