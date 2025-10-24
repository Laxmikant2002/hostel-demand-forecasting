# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, visualization, and experimentation.

## 📓 Notebook Organization

```
notebooks/
├── 01_data_exploration.ipynb       # Initial EDA on hotel bookings
├── 02_airbnb_analysis.ipynb        # Airbnb data exploration
├── 03_data_cleaning.ipynb          # Data cleaning and preprocessing
├── 04_feature_engineering.ipynb    # Feature creation and selection
├── 05_time_series_analysis.ipynb   # Time series patterns and trends
├── 06_forecasting_models.ipynb     # Model development and testing
├── 07_model_evaluation.ipynb       # Model comparison and metrics
└── 08_visualization.ipynb          # Final visualizations
```

## 🎯 Purpose

Use notebooks for:
- **Exploration**: Understand data patterns and distributions
- **Visualization**: Create charts and graphs
- **Prototyping**: Test ideas and algorithms quickly
- **Documentation**: Show analysis process with narrative

## 📝 Best Practices

1. **Naming Convention**: Use numbered prefixes (01_, 02_, etc.) to show workflow order
2. **Clear Titles**: Descriptive names that explain notebook purpose
3. **Documentation**: Add markdown cells explaining each step
4. **Clean Output**: Clear outputs before committing to Git
5. **Modular Code**: Move reusable code to `src/` directory

## 🔄 Workflow

1. **Start Here**: Begin analysis in notebooks
2. **Iterate**: Experiment with different approaches
3. **Refine**: Once code works, move to scripts in `src/`
4. **Document**: Keep notebooks as documentation of analysis

## 🚀 Quick Start

### Create New Notebook
```bash
# In VS Code: Ctrl+Shift+P -> "Create: New Jupyter Notebook"
# Or from terminal:
jupyter notebook
```

### Load Data in Notebook
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load hotel data
hotel_df = pd.read_csv('../data/raw/hotel_bookings.csv')

# Quick exploration
hotel_df.head()
```

## 📚 Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

---

**Status**: Ready for analysis notebooks  
**Next**: Create `01_data_exploration.ipynb`
