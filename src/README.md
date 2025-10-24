# Source Code Directory

This directory contains reusable Python modules and scripts for the hostel demand forecasting project.

## 📁 Directory Structure

```
src/
├── __init__.py                 # Package initialization
├── data/
│   ├── __init__.py
│   ├── data_loader.py          # Load data from various sources
│   ├── data_cleaner.py         # Data cleaning functions
│   └── data_preprocessor.py    # Preprocessing and feature engineering
├── models/
│   ├── __init__.py
│   ├── prophet_model.py        # Facebook Prophet implementation
│   ├── arima_model.py          # ARIMA/SARIMA models
│   ├── ml_models.py            # Machine learning models
│   └── model_evaluator.py      # Model evaluation metrics
├── visualization/
│   ├── __init__.py
│   ├── time_series_plots.py    # Time series visualizations
│   └── dashboard.py            # Streamlit dashboard components
└── utils/
    ├── __init__.py
    ├── config.py               # Configuration settings
    └── helpers.py              # Utility functions
```

## 🎯 Purpose

The `src/` directory contains:
- **Reusable Functions**: Code used across multiple notebooks/scripts
- **Model Implementations**: Forecasting algorithms
- **Data Pipeline**: ETL and preprocessing logic
- **Utilities**: Helper functions and configurations

## 📝 Coding Standards

### Module Structure
```python
"""
Module description.

This module contains...
"""

import pandas as pd
import numpy as np

def function_name(param1, param2):
    """
    Brief description.
    
    Args:
        param1 (type): Description
        param2 (type): Description
    
    Returns:
        type: Description
    
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    # Implementation
    return result
```

### Best Practices
1. **Type Hints**: Use type annotations
2. **Docstrings**: Document all functions and classes
3. **Error Handling**: Use try-except blocks
4. **Testing**: Write unit tests (in `tests/` directory)
5. **PEP 8**: Follow Python style guidelines

## 🔄 Workflow

### Development Flow
1. **Prototype** in Jupyter notebooks
2. **Extract** working code to functions
3. **Organize** into appropriate modules
4. **Test** functionality
5. **Document** with docstrings
6. **Import** in notebooks/scripts

### Example Usage
```python
# In a notebook or script
from src.data.data_loader import load_hotel_data
from src.data.data_cleaner import clean_hotel_data
from src.models.prophet_model import train_prophet_model

# Use the functions
df = load_hotel_data()
df_clean = clean_hotel_data(df)
model = train_prophet_model(df_clean)
```

## 📦 Package Installation

To use `src` as a package:
```bash
# From project root
pip install -e .
```

This requires a `setup.py` file in the project root.

## 🧪 Testing

Place tests in `tests/` directory (to be created):
```
tests/
├── test_data_loader.py
├── test_data_cleaner.py
└── test_models.py
```

Run tests:
```bash
pytest tests/
```

## 📚 Module Guidelines

### data/data_loader.py
- Load data from various sources
- Handle different file formats
- Data validation

### data/data_cleaner.py
- Handle missing values
- Remove duplicates
- Data type conversions
- Outlier detection

### data/data_preprocessor.py
- Feature engineering
- Encoding categorical variables
- Scaling/normalization
- Train-test splitting

### models/
- Implement forecasting algorithms
- Model training functions
- Prediction functions
- Model serialization (save/load)

### visualization/
- Reusable plotting functions
- Dashboard components
- Interactive visualizations

### utils/
- Configuration management
- File path helpers
- Date/time utilities
- Logging setup

## 🚀 Quick Start

### Create __init__.py files
```python
# src/__init__.py
"""Hostel Demand Forecasting Package"""
__version__ = '0.1.0'

# src/data/__init__.py
from .data_loader import load_hotel_data, load_airbnb_data
from .data_cleaner import clean_hotel_data, clean_airbnb_data

__all__ = ['load_hotel_data', 'load_airbnb_data', 
           'clean_hotel_data', 'clean_airbnb_data']
```

### Example Module (data_loader.py)
```python
"""Data loading utilities."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

def load_hotel_data(filepath=None):
    """
    Load hotel bookings data.
    
    Args:
        filepath (str, optional): Custom file path
    
    Returns:
        pd.DataFrame: Hotel bookings data
    """
    if filepath is None:
        filepath = DATA_DIR / 'raw' / 'hotel_bookings.csv'
    
    return pd.read_csv(filepath)
```

## 🔧 Configuration

Create `src/utils/config.py`:
```python
"""Configuration settings."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

# Model parameters
PROPHET_PARAMS = {
    'seasonality_mode': 'multiplicative',
    'yearly_seasonality': True,
    'weekly_seasonality': True
}

# Random seed for reproducibility
RANDOM_SEED = 42
```

---

**Status**: Directory structure ready  
**Next Steps**:
1. Create `__init__.py` files
2. Move reusable code from notebooks to modules
3. Write unit tests
4. Document all functions
