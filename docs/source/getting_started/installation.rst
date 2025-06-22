.. _installation:

Installation
============

C60.ai requires Python 3.8 or higher. The recommended way to install C60.ai is using pip.

Stable Release
-------------

To install the latest stable release of C60.ai, run:

.. code-block:: bash

    pip install c60-ai

From Source
-----------

If you want to install C60.ai from source, follow these steps:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/aditirkrishna/c60.ai.git
       cd c60.ai

2. Install the package in development mode with all dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

   This will install the package in development mode along with all development dependencies.

Dependencies
------------

C60.ai has the following core dependencies:

- Python 3.8+
- NumPy
- pandas
- scikit-learn
- NetworkX
- Optuna
- MLflow
- FastAPI
- uvicorn
- pydantic
- pyarrow
- matplotlib
- seaborn
- plotly
- xgboost
- lightgbm
- catboost

All dependencies will be automatically installed when installing via pip.

Verifying the Installation
--------------------------

To verify that C60.ai has been installed correctly, you can run the following command in a Python shell:

.. code-block:: python

    import c60
    print(f"C60.ai version: {c60.__version__}")

If the installation was successful, this should print the version of C60.ai that was installed.

Troubleshooting
---------------

If you encounter any issues during installation, please check the following:

1. Ensure you have Python 3.8 or higher installed.
2. Make sure you have the latest version of pip:

   .. code-block:: bash

       pip install --upgrade pip

3. If you're installing from source, ensure you have a C/C++ compiler installed and the Python development headers.
4. If you're using a virtual environment, make sure it's activated before installing.

If you're still having trouble, please `open an issue <https://github.com/aditirkrishna/c60.ai/issues>`_ with details about your environment and the error message you're seeing.
