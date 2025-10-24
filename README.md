# 🏨 Hostel Demand Forecasting

Time series forecasting project for hotel booking demand using machine learning and statistical models.

## 📚 Project Overview

This project analyzes and forecasts hotel booking demand using the **Hotel Booking Demand Dataset** from Kaggle. The dataset contains 119,390 bookings from City and Resort hotels between July 2015 and August 2017.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+ (Conda environment recommended)
- VS Code
- Kaggle account

### 2. Setup Instructions

#### Install Required Libraries

All necessary Python libraries have been installed:
- ✅ Core data science: `pandas`, `numpy`, `matplotlib`, `seaborn`
- ✅ Forecasting models: `prophet`, `statsmodels`, `scikit-learn`
- ✅ MongoDB connection: `pymongo`, `dnspython`
- ✅ Data collection: `kaggle`, `requests`, `beautifulsoup4`
- ✅ Visualization & Dashboard: `plotly`, `streamlit`
- ✅ Utilities: `opendatasets`

#### Download Dataset

**Step 1: Set up Kaggle API credentials**

1. Go to [https://www.kaggle.com/](https://www.kaggle.com/) and sign in
2. Navigate to: Profile → Account → API → "Create New API Token"
3. Download `kaggle.json`
4. Place it in: `C:\Users\laxmi\.kaggle\kaggle.json` (folder already created)

**Step 2: Download the dataset**

Run the download script:
```bash
python download_kaggle_dataset.py
```

**Step 3: Verify the download**

```bash
python verify_dataset.py
```

For detailed instructions, see [KAGGLE_DOWNLOAD_GUIDE.md](KAGGLE_DOWNLOAD_GUIDE.md)

## 📂 Project Structure

```
hostel-demand-forecasting/
├── data/                          # All datasets
│   ├── raw/                       # Original, immutable data
│   │   ├── hotel_bookings.csv     # Kaggle dataset (119K records)
│   │   └── airbnb/                # InsideAirbnb data
│   │       └── prague/            # Prague Airbnb data (9.4K listings)
│   │           ├── listings.csv   # Property details
│   │           ├── calendar.csv   # Daily availability
│   │           └── reviews.csv    # Guest reviews
│   ├── processed/                 # Cleaned and transformed data
│   ├── external/                  # External data (events, holidays, weather)
│   └── README.md                  # Data documentation
│
├── notebooks/                     # Jupyter notebooks for analysis
│   └── README.md                  # Notebook guidelines
│
├── src/                           # Source code modules
│   ├── data/                      # Data loading and preprocessing
│   ├── models/                    # Forecasting models
│   ├── visualization/             # Plotting and dashboard
│   ├── utils/                     # Utility functions
│   └── README.md                  # Code documentation
│
├── 01_data_exploration.py         # Initial data exploration script
├── 02_download_airbnb.py          # Airbnb data downloader
├── 03_explore_airbnb_data.py      # Airbnb data exploration
├── download_kaggle_dataset.py     # Kaggle dataset downloader
├── verify_dataset.py              # Dataset verification
├── setup_kaggle.py                # Kaggle API setup helper
├── .env                           # Environment variables
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 📊 Dataset Details

- **Source**: [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **Records**: 119,390 bookings
- **Features**: 32 columns
- **Hotels**: City Hotel & Resort Hotel
- **Time Period**: July 2015 - August 2017
- **Key Features**:
  - Hotel type
  - Arrival date (year, month, week, day)
  - Lead time
  - Number of guests
  - Average Daily Rate (ADR)
  - Cancellation status
  - Special requests
  - Customer type

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Prophet**: Time series forecasting
- **Statsmodels**: Statistical modeling
- **Scikit-learn**: Machine learning
- **MongoDB**: Database storage
- **Streamlit**: Interactive dashboard
- **Plotly**: Interactive visualizations

## 📈 Project Goals

1. **Exploratory Data Analysis (EDA)**: Understand booking patterns and trends
2. **Time Series Forecasting**: Predict future hotel demand
3. **Cancellation Prediction**: Identify factors influencing cancellations
4. **Price Optimization**: Analyze ADR patterns
5. **Dashboard Creation**: Build interactive visualizations

## 🎯 Next Steps

1. ✅ Install Python libraries
2. ⏳ Download Kaggle dataset (in progress)
3. ⏳ Exploratory Data Analysis
4. ⏳ Data preprocessing and feature engineering
5. ⏳ Build forecasting models
6. ⏳ Model evaluation and comparison
7. ⏳ Create interactive dashboard
8. ⏳ Deploy to production

## 📝 Notes

- The `.kaggle` folder has been created at `C:\Users\laxmi\.kaggle\`
- Remember to place your `kaggle.json` API token in this folder
- Run `download_kaggle_dataset.py` after setting up Kaggle credentials
- Use `verify_dataset.py` to confirm successful download

## 📞 Support

For issues with:
- **Kaggle API**: See [KAGGLE_DOWNLOAD_GUIDE.md](KAGGLE_DOWNLOAD_GUIDE.md)
- **Dataset**: Run `verify_dataset.py` for diagnostics
- **Libraries**: Check installed packages with `pip list`

---

**Happy Forecasting! 🚀**