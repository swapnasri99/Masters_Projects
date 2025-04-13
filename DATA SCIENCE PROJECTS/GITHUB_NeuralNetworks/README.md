# 🧠 Neural Networks Lab 2 Project – PyTorch

This project explores how to build, train, and evaluate neural networks using **PyTorch** through two main tasks:

1. **Fashion MNIST – Classification**
2. **California Housing – Regression**

The notebook includes full data pipelines, training loops, evaluation functions, and model exporting. It also demonstrates the use of advanced training tools like schedulers and early stopping.

---

## 📦 Project Structure

```
📦 Neural Networks Project
┣ 📜 HomeworkVersion_Lab2_.ipynb   # Main project notebook (Classification + Regression)
┣ 📜 data.zip                      # Contains the California Housing dataset (unzipped to data/task5/)
┣ 📜 submission.csv                # Final predictions for regression task
┗ 📜 README.md                     # This file
```

-
## 📌 Tasks Overview (Elaborated)

### 🔹 1. Fashion MNIST – Image Classification

This part focuses on building and training a fully connected neural network (MLP) to classify grayscale images from the Fashion MNIST dataset (e.g., shirts, shoes, bags, etc.).

#### ✅ Key Steps:

- Used **PyTorch's `nn.Module`** and `nn.Sequential` to define MLP architectures.
- Applied layers:
  - Input → ReLU → Dropout
  - Hidden → ReLU → Dropout
  - Output → Softmax
- Switched from **SGD** to **Adam optimizer** for better training dynamics.
- Integrated a **Linear Learning Rate Scheduler (`LinearLR`)** to linearly reduce learning rate from `1.0 × lr` to `0.33 × lr` over the first 10 epochs.
- Implemented **early stopping** to prevent overfitting by monitoring validation loss with a patience of 5 epochs.
- Used **macro F1 score** to evaluate multi-class classification performance.
- Model training used **PyTorch’s Dataset API**:
  - Data loaded using `TensorDataset` and `DataLoader`
  - Batching and shuffling handled efficiently for GPU training
- `torchsummary.summary()` used to visualize model architecture and total parameters.

---

### 🔹 2. California Housing – Regression Task

In this task, a regression neural network is built to predict **house prices** using structured tabular data from the California Housing dataset.

#### ✅ Key Steps:

- Normalized input features using **`StandardScaler`** for stable neural network training.
- Split data using `train_test_split` into training and validation sets (80/20).
- Built a custom **regression MLP model (`RegressionNetwork`)**:
  - Input layer → ReLU → Hidden layers → Linear output (1 unit)
- Used `MSELoss` as the training criterion for continuous value prediction.
- Used **`TensorDataset`** to wrap input and target tensors together.
- Used **`DataLoader`** to:
  - Efficiently feed data in mini-batches
  - Enable shuffling for the training set
  - Ensure batch-wise evaluation during validation
- Reshaped target tensors using `.unsqueeze(1)` to match `[batch_size, 1]` output shape.
- Added **early stopping** with patience = 10 to restore best weights and halt overfitting.
- Applied `LinearLR` learning rate scheduler across 10 epochs.
- Saved model predictions on test data to `submission.csv`.
- Evaluated using:
  - ✅ Mean Squared Error (MSE)
  - ✅ R² Score (variance explained by the model)

---


## 🧪 Model Training Pipeline

### 🔁 Classification

```python
def epoch()         # Trains the model for one epoch
def evaluate()      # Computes loss + F1 score on validation data
def fit()           # Full training loop with early stopping + scheduler
```

### 🔁 Regression

```python
def epoch_func()      # Trains for one epoch
def evaluate_func()   # Validates and returns predictions + loss
def fit_func()        # Full training loop for regression with early stopping + scheduler
```

### 🔍 Tools Used

- `loss_fn(y_pred, y_batch)` with `.item()` or `.detach().cpu().numpy()` for safe logging
- `y_batch.unsqueeze(1)` to reshape labels for regression
- `with torch.no_grad()` during inference to reduce memory usage
- `model.state_dict()` to store and restore best model weights

---

## 🧾 Key Features

- ✅ `Adam` optimizer for better convergence
- ✅ `LinearLR` scheduler: adjusts LR from 1.0× to 0.33× over 10 epochs
- ✅ `Early Stopping`: monitors validation loss and restores best model
- ✅ `F1 Score` for classification, `MSE`, `R²` for regression

---

## 📤 Exporting Predictions

```python
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    predictions = model(X_test_tensor)

pd.DataFrame(predictions.cpu().numpy(), columns=['predictions']).to_csv('submission.csv', index=False)
```

---

## 📈 Results Evaluation

- Fashion MNIST F1 Score: consistently improved across epochs
- Regression MSE: meets required threshold for model acceptance
- Model saved and exported for deployment

---

## 📌 How to Run This Project (Colab Recommended)

1. Open `HomeworkVersion_Lab2_.ipynb` in Google Colab
2. Upload `data.zip` and run the extraction cell:
   ```python
   with zipfile.ZipFile('data.zip', 'r') as zip_ref:
       zip_ref.extractall()
   ```
3. Train classification and regression models
4. Evaluate and generate predictions
5. Download `submission.csv` for submission

---


## 🧠 Built With

- Python 3.x  
- PyTorch  
- NumPy & Pandas  
- Scikit-learn  
- Google Colab / Jupyter

---

End