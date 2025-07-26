# ML Project

This project is designed to implement various machine learning models for fraud detection using a dataset loaded from an Excel file. It includes functionalities for data loading, preprocessing, model training, and evaluation.

## Project Structure

```
ml-project
├── src
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model_runner.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   └── test_model_runner.py
├── data
│   └── .gitkeep
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Place your Excel dataset in the `data` directory.
2. Modify the `src/main.py` file to specify the path to your dataset and the target column.
3. Run the main module:

```
python src/main.py
```

## Testing

To run the tests, use:

```
pytest
```

## License

This project is licensed under the MIT License.

# ML Project Example

This project demonstrates a complete machine learning pipeline with different models.

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

## Running the Example

1. Make sure you're in the project root directory:

```bash
cd path/to/ml-project
```

2. Run the complete example:

```bash
python examples/complete_example.py
```

The example will:

- Create a synthetic dataset
- Train multiple models (Random Forest, Logistic Regression, SVM)
- Compare model performance
- Save results to the data directory

## Project Structure

```
ml-project/
├── src/             # Core modules
├── tests/           # Unit tests
├── examples/        # Example scripts
├── data/           # Data directory
└── requirements.txt # Dependencies
```
