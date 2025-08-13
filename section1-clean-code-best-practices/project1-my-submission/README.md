# Predict Customer Churn with MLOps Best Practices

## Project Overview

This project addresses the critical business problem of **customer churn prediction** using machine learning, with a strong emphasis on **clean code principles, software engineering best practices, and foundational MLOps concepts**. It provides a robust framework for identifying customers most likely to churn, enabling businesses to implement targeted retention strategies. The solution is built with modularity, testability, and maintainability in mind, showcasing a production-ready approach to data science projects.

## Features

*   **Customer Churn Prediction:** Develops and deploys a machine learning model to predict customer churn.
*   **Clean Code & Modularity:** Adheres strictly to PEP 8 guidelines and implements a modular codebase (`churn_library.py`) for reusability and clarity.
*   **Automated Testing:** Includes comprehensive unit tests (`churn_script_logging_and_tests.py`) to ensure code correctness and reliability.
*   **Robust Logging:** Implements structured logging to monitor application flow, debug issues, and track model performance.
*   **End-to-End ML Workflow:** Covers data ingestion, EDA, feature engineering, model training (Logistic Regression, Random Forest), evaluation, and persistence.
*   **Data Visualization:** Generates insightful plots for EDA and model performance analysis, saved as image assets.
*   **Model Persistence:** Saves trained models (`logistic_model.pkl`, `rfc_model.pkl`) for future use and deployment.

## Tech Stack

*   **Python:** Core programming language (developed with Python 3.10.1).
*   **Pandas:** For efficient data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Scikit-learn:** For machine learning model development, training, and evaluation.
*   **Matplotlib / Seaborn:** For creating static, animated, and interactive visualizations.
*   **Joblib / Pickle:** For model serialization and deserialization.
*   **Logging:** Python's built-in logging module for application monitoring.
*   **Jupyter Notebooks:** For interactive development and experimentation.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/udacity-machine-learning-devops-engineer.git
    cd udacity-machine-learning-devops-engineer/section1-clean-code-best-practices/project1-my-submission
    ```
    *(Note: Replace `your-username` with the actual GitHub username if this project is publicly hosted.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib
    ```
    *(Note: A `requirements.txt` file is present in the repository, but its content could not be read. The above command lists common dependencies. Please refer to the original `requirements.txt` if possible for exact versions.)*

## Usage

### Running the Main Script and Tests

The `churn_script_logging_and_tests.py` script orchestrates the entire ML pipeline, including data loading, preprocessing, model training, evaluation, and logging. It also contains unit tests for the `churn_library.py` functions.

To run the full pipeline and execute tests:

```bash
python src/churn_script_logging_and_tests.py
```

This script will:
*   Load `bank_data.csv` from the `data/` directory.
*   Perform EDA and save visualizations to `images/eda/`.
*   Train Logistic Regression and Random Forest models.
*   Evaluate models and save performance plots to `images/results/`.
*   Save trained models to `models/`.
*   Generate log messages in `logs/`.
*   Execute unit tests for `churn_library.py`.

### Interactive Development and Exploration

The `notebooks/` directory contains the Jupyter Notebook used for initial development and interactive exploration.

*   `notebooks/churn_notebook.ipynb`: Provides a step-by-step walkthrough of the data analysis, feature engineering, and model building process.

To run this notebook, ensure you have Jupyter installed (`pip install jupyter`) and then:
```bash
jupyter lab notebooks/churn_notebook.ipynb
```

## Project Structure

```
.
├── data/
│   └── bank_data.csv
├── images/
│   ├── eda/
│   └── results/
├── logs/
│   ├── churn_library.log
│   └── churn_script_logging_and_tests.log
├── models/
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── notebooks/
│   └── churn_notebook.ipynb
├── src/
│   ├── churn_library.py
│   └── churn_script_logging_and_tests.py
├── .git/
├── README.md
└── requirements.txt
```

*   `data/`: Stores the raw dataset (`bank_data.csv`).
*   `images/`: Contains subdirectories for EDA visualizations (`eda/`) and model performance results (`results/`).
*   `logs/`: Stores application logs for debugging and monitoring.
*   `models/`: Contains serialized machine learning models (`.pkl` files).
*   `notebooks/`: Houses the Jupyter Notebook used for interactive development.
*   `src/`: Contains the core Python scripts, including the modular library and the main script with tests.
*   `README.md`: This file, providing an overview and instructions.
*   `requirements.txt`: Lists project dependencies.

## Onboarding for New Developers

Welcome to the project! Here's how to get started and contribute effectively:

1.  **Familiarize yourself with the project overview and usage instructions.**
2.  **Explore `src/churn_library.py`:** Understand the modular functions for data loading, preprocessing, feature engineering, training, and evaluation.
3.  **Review `src/churn_script_logging_and_tests.py`:** Understand the end-to-end pipeline and how tests are integrated.
4.  **Examine `notebooks/churn_notebook.ipynb`:** Gain insights into the iterative development process and data characteristics.
5.  **Code Style:** Strictly adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for all Python code. Use linters (e.g., `flake8`, `pylint`) to maintain code quality.
6.  **Testing:** Write unit tests for any new functions or significant changes. Ensure existing tests pass before submitting contributions.
7.  **Logging:** Utilize the logging module for informative messages, especially for critical operations and error handling.
8.  **Version Control:** Use Git for version control. Create new branches for features or bug fixes and submit Pull Requests.
9.  **Documentation:** Keep comments concise and explain *why* certain decisions were made, especially for complex logic or non-obvious implementations.

## For Recruiters

### Architecture Overview

This project exemplifies a well-structured **Machine Learning Operations (MLOps)** approach, demonstrating a clear separation of concerns and adherence to software engineering principles throughout the ML lifecycle.

*   **Data Layer (`data/`):** Centralized and versioned storage for the raw dataset, ensuring data integrity and reproducibility.
*   **Processing & Feature Engineering Layer (`src/churn_library.py`):** Modular functions for data preprocessing and feature engineering, promoting reusability and testability.
*   **Modeling Layer (`src/churn_library.py`, `models/`):** Encapsulated model training and evaluation logic, with trained models persisted for deployment readiness.
*   **Orchestration & Testing Layer (`src/churn_script_logging_and_tests.py`):** A single entry point for running the entire ML pipeline, including automated unit tests and comprehensive logging, crucial for CI/CD integration and operational monitoring.
*   **Visualization & Reporting Layer (`images/`, `logs/`):** Automated generation of visual insights and detailed logs, providing transparency and traceability for model performance and application behavior.

This architecture is designed for maintainability, scalability, and ease of integration into larger MLOps pipelines, showcasing a mature approach to building and deploying ML solutions.

### Key Learnings & Expertise Demonstrated

*   **MLOps Principles:** Applied core MLOps concepts including modularity, testing, logging, and model versioning, preparing the project for production deployment.
*   **Clean Code & Software Engineering:** Developed highly readable, maintainable, and testable Python code adhering to industry best practices (PEP 8).
*   **Machine Learning Lifecycle:** End-to-end experience from data ingestion and EDA to model training, evaluation, and persistence for both Logistic Regression and Random Forest models.
*   **Automated Testing:** Implemented and utilized unit tests to ensure the reliability and correctness of the ML pipeline components.
*   **Robust Logging:** Designed and integrated a comprehensive logging strategy for effective monitoring and debugging of the application.
*   **Data Visualization:** Proficient in generating insightful visualizations for exploratory data analysis and communicating model performance.
*   **Version Control (Git):** Experience with standard Git workflows for collaborative development and project management.

### Future Improvements

*   **CI/CD Pipeline:** Implement a Continuous Integration/Continuous Deployment pipeline (e.g., using GitHub Actions, Jenkins) to automate testing, model retraining, and deployment.
*   **Model Monitoring:** Integrate tools for continuous monitoring of model performance in production (e.g., drift detection, performance degradation alerts).
*   **API Deployment:** Expose the churn prediction model as a RESTful API (e.g., using Flask, FastAPI) for easy integration with other applications.
*   **Containerization:** Containerize the application using Docker for consistent deployment across different environments.
*   **Experiment Tracking:** Integrate experiment tracking tools (e.g., MLflow, Weights & Biases) to manage model versions, parameters, and metrics.
*   **Advanced Modeling:** Explore more advanced machine learning techniques (e.g., gradient boosting, neural networks) and ensemble methods for improved prediction accuracy.
*   **Data Versioning:** Implement data versioning (e.g., DVC) to track changes in datasets and ensure reproducibility.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries or collaborations, please contact [Your Name/Email/LinkedIn Profile].
