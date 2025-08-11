Project-2/
│
├── data/
│   └── Data.xlsx                # Original raw dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Constants and global configs
│   ├── data_loader.py           # Load and preprocess data
│   ├── feature_engineering.py   # Aggregation and feature cleanup
│   ├── model/
│   │   ├── train_model.py       # Model training (ETR, LR)
│   │   ├── evaluate.py          # Evaluation metrics
│   │   └── optimization.py      # KOA & COA optimizers
│   ├── analysis/
│   │   ├── sensitivity.py       # Copula-based sensitivity analysis
│   │   ├── wilcoxon.py          # Wilcoxon Test implementation
│   │   └── runtime.py           # Runtime reporting
│   └── utils.py                 # Helper functions
│
├── notebooks/                   # Jupyter notebooks (for exploratory tasks)
│   └── eda.ipynb
│
├── results/                     # Outputs (metrics, plots, etc.)
│
├── requirements.txt             # All packages used
└── main.py                      # Pipeline runner
