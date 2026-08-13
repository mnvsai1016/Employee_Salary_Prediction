# 💼 Tech Employee Salary Predictor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-EB5424.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning web application built with **Streamlit**, **Pandas**, **Scikit-Learn**, and **XGBoost** to predict global software developer and tech employee salaries based on demographics, experience, job roles, and technical skill stacks.

---

## 🌟 Key Features

- **🎨 Modern Tabbed Interface**: Structured step-by-step survey layout divided into Demographics, Tech Stack, and Developer Tools.
- **⚡ Fast Inference with Caching**: Resource-cached model loading using `@st.cache_resource` for instant salary estimations without performance lag.
- **🛡️ Built-in Input Validation**: Dynamic form validation ensuring complete profile inputs before invoking model prediction.
- **📊 Detailed Salary Ranges**: Displays a median annual salary estimate alongside an estimated range ($\pm 10\%$) with high-contrast UI metrics.
- **🔧 Automated Feature Alignment**: Robust preprocessing pipeline transforming multi-label lists, ordinal categories, and log-scaled experience with feature reindexing.
- **🐳 Docker Containerized**: Ready for containerized deployment across cloud providers (AWS, GCP, Azure, Render).

---

## 📂 Project Architecture & Directory Structure

```
Employee_Salary_Prediction/
├── main.py                # Streamlit web application frontend & prediction logic
├── preprocess.py          # Data preprocessing, feature engineering & alignment pipeline
├── requirements.txt       # Project dependencies (Streamlit, Scikit-Learn, XGBoost, Pandas)
├── Dockerfile             # Docker container definition for production deployment
├── .gitignore             # Git ignore definitions
├── testingfile.ipynb      # EDA, feature analysis, and model training notebook
├── model_files.zip        # Compressed archive of trained model artifacts
├── model_files/           # Extracted model artifacts directory
│   ├── salary_model.pkl   # Pre-trained ML model (XGBoost/Scikit-Learn estimator)
│   ├── country_encoder.pkl# LabelEncoder for Country classification
│   ├── features.pkl       # Feature column alignment list
│   └── scaler.pkl         # Feature scaling parameters
└── README.md              # Project documentation
```

---

## 🛠️ Data Preprocessing & Machine Learning Pipeline

```
Raw Survey Input
       │
       ├── Ordinal Encoding (Age, Education Level)
       ├── Label Encoding (Country with unseen label fallback)
       ├── One-Hot Encoding (Job Role / DevType, Work Setup)
       ├── Multi-Hot Binarization (Languages, Databases, Cloud, Frameworks, Tools)
       └── Log Transformation (Work Experience log1p)
       │
       ▼
Feature Alignment (reindex against features.pkl)
       │
       ▼
XGBoost Regression Model (salary_model.pkl)
       │
       ▼
Inverse Log Transform (np.expm1) -> Estimated Annual Salary ($ USD)
```

---

## 🚀 Local Setup & Quick Start

### 1. Clone & Navigate to Project

```bash
cd Employee_Salary_Prediction
```

### 2. Install Dependencies

Install all required packages via `pip`:

```bash
pip install -r requirements.txt
```

### 3. Extract Model Artifacts

Ensure `model_files.zip` is extracted into the `model_files/` directory in the project root:

```bash
# Windows PowerShell
Expand-Archive -Path "model_files.zip" -DestinationPath "." -Force
```

### 4. Launch the Streamlit App

Run the application locally:

```bash
streamlit run main.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended & Free)

1. Push your repository (including `model_files/` and `requirements.txt`) to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your GitHub repository and branch.
4. Set the **Main file path** to `main.py`.
5. Click **"Deploy!"**. Your app will be live on a public URL in ~1 minute.

### Option 2: Docker Container (Any Cloud / Server)

Build and run the Docker image locally or on a cloud instance (AWS EC2, GCP Cloud Run, DigitalOcean, Azure):

```bash
# 1. Build Docker image
docker build -t employee-salary-predictor .

# 2. Run container on port 8501
docker run -d -p 8501:8501 --name salary-app employee-salary-predictor
```

Access the app at `http://localhost:8501`.

### Option 3: Hugging Face Spaces (Free hosting)

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Push your repository files. Hugging Face will automatically build and host the app.

---

## 🧪 Model Exploration & Notebook

For model training details, feature engineering experiments, and Exploratory Data Analysis (EDA), check out [`testingfile.ipynb`](file:///C:/Users/Mnvsai/Desktop/Github/Employee_Salary_Prediction-main/Employee_Salary_Prediction/testingfile.ipynb).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
