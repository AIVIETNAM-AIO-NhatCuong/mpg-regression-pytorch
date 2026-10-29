# 🚗 Auto MPG Fuel Efficiency Prediction with PyTorch

A deep learning regression pipeline built with **PyTorch** to predict automobile fuel efficiency (Miles Per Gallon) using vehicle features such as cylinders, displacement, horsepower, weight, acceleration, model year, and origin.

---

## 📌 Project Overview

This repository demonstrates an end-to-end Machine Learning pipeline for tabular regression using PyTorch. The goal is to build a Multi-Layer Perceptron (MLP) neural network capable of predicting the fuel economy (`MPG`) of vehicles based on technical specifications.

### Key Features
- **Data Preprocessing & Normalization:** Standardized numerical features using `scikit-learn`'s `StandardScaler`.
- **Custom PyTorch Dataset & DataLoader:** Batch processing using PyTorch primitives.
- **Custom Regression Loss & Metrics:** Evaluation using Mean Squared Error (`nn.MSELoss`) and a custom PyTorch-native **$R^2$ (Coefficient of Determination)** metric.
- **Full Model Lifecycle:** Structured training and evaluation loops with per-epoch metric tracking.

---

## 📊 Dataset

The dataset used in this project is the classic **Auto MPG Dataset**.

* **Dataset Size:** 392 instances (after handling missing values)
* **Target Variable:** `MPG` (Continuous float)
* **Input Features (9 total):**
  * `Cylinders`, `Displacement`, `Horsepower`, `Weight`, `Acceleration`, `Model Year`
  * One-Hot Encoded Origin (`Europe`, `Japan`, `USA`)

### Train / Validation / Test Split
| Split | Percentage | Number of Samples |
| :--- | :--- | :--- |
| **Train** | 70% | 281 |
| **Validation** | 20% | 79 |
| **Test** | 10% | 32 |

---

## 🏗 Model Architecture

A fully connected **Multi-Layer Perceptron (MLP)** with non-linear activation functions:

```
Input Layer (9 Features)
       ↓
Dense / Linear (9 → 64)  +  ReLU Activation
       ↓
Dense / Linear (64 → 64) +  ReLU Activation
       ↓
Output Layer (64 → 1)   → Continuous MPG Prediction
```

* **Optimizer:** Stochastic Gradient Descent (SGD) with `lr = 0.01`
* **Loss Function:** Mean Squared Error (`nn.MSELoss()`)
* **Training Duration:** 100 Epochs

---

## 📈 Training Results

The model achieved strong convergence over 100 training epochs, demonstrating high predictive accuracy without overfitting.

![Training Results](https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME/main/training_results.png)

*(Note: Replace the image link above with your actual uploaded image path or repository asset path)*

### Performance Metrics
* **Final Training $R^2$ Score:** `0.8960`
* **Final Validation $R^2$ Score:** `0.9145`

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib torch scikit-learn gdown
```

### 3. Run the Notebook or Script
Open `Auto_MPG_Prediction.ipynb` in Google Colab, Jupyter Notebook, or VS Code and execute all cells.

---

## 🛠 Tech Stack

- **Framework:** PyTorch
- **Data Analysis & Processing:** Pandas, NumPy, Scikit-Learn
- **Visualization:** Matplotlib

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).