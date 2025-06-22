.. _quickstart:

Quickstart
==========

This guide will help you get started with C60.ai by walking you through a simple example of building and optimizing a machine learning pipeline.

Prerequisites
------------

- Python 3.8 or higher
- C60.ai installed (see :ref:`installation`)
- Basic knowledge of machine learning concepts

Your First Pipeline
-------------------

Let's create a simple machine learning pipeline using C60.ai:

.. code-block:: python

    from c60 import AutoML
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize AutoML
    automl = AutoML(
        task='classification',
        time_budget=60,  # 1 minute
        metric='accuracy',
        n_jobs=-1  # Use all available cores
    )

    # Fit the model
    automl.fit(X_train, y_train)

    # Make predictions
    predictions = automl.predict(X_test)

    probabilities = automl.predict_proba(X_test)


    # Evaluate the model
    score = automl.score(X_test, y_test)
    print(f"Model accuracy: {score:.4f}")

Advanced Usage
-------------

### Customizing the Search Space

You can customize the search space for the AutoML process:

.. code-block:: python

    from c60 import AutoML
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # Define custom search space
    search_space = {
        'preprocessor': [
            ('scale', StandardScaler()),
            ('minmax', MinMaxScaler()),
            None
        ],
        'classifier': [
            ('rf', RandomForestClassifier()),
            ('gbm', GradientBoostingClassifier()),
            ('svm', SVC(probability=True))
        ]
    }

    automl = AutoML(
        task='classification',
        search_space=search_space,
        time_budget=300,  # 5 minutes
        metric='f1',
        n_jobs=-1
    )

### Hyperparameter Optimization

C60.ai uses Optuna for hyperparameter optimization. You can customize the optimization process:

.. code-block:: python

    from c60 import AutoML
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner

    automl = AutoML(
        task='classification',
        time_budget=300,
        metric='roc_auc',
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner(),
        n_trials=100,
        n_jobs=-1
    )

### Cross-Validation

Use cross-validation for more robust model evaluation:

.. code-block:: python

    from c60 import AutoML
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    automl = AutoML(
        task='classification',
        time_budget=300,
        cv=cv,
        metric='accuracy',
        n_jobs=-1
    )

Saving and Loading Models
------------------------

You can save and load trained models:

.. code-block:: python

    # Save the model
    automl.save('automl_model.joblib')

    # Load the model
    from c60 import AutoML
    automl_loaded = AutoML.load('automl_model.joblib')

    # Make predictions with the loaded model
    predictions = automl_loaded.predict(X_test)

Next Steps
----------

- Learn more about the :ref:`AutoML class <api/auto_generated/c60.automl>`
- Explore more :ref:`examples <getting_started/examples/index>`
- Read the :ref:`user guide <user_guide/overview>` for advanced usage
- Check out the :ref:`API reference <api/auto_generated/modules>` for detailed documentation
