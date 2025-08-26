# 🖊️ Digit Recognizer

A simple machine learning project that recognizes handwritten digits (0–9) using the **digits dataset** from `scikit-learn`.  
The model is built using **Support Vector Machines (SVM)** with preprocessing, and achieves **~98–99% accuracy**.

<img width="1500" height="1200" alt="image" src="https://github.com/user-attachments/assets/3e725379-253d-44e9-b252-8ffb434ed194" />


---

## 📂 Project Structure
```

digitrecognizer/
├── artifacts/                 # generated after training
│   ├── model.joblib           # saved model
│   ├── metrics.txt            # accuracy + classification report
│   ├── confusion\_matrix.png   # confusion matrix visualization
│   ├── samples.png            # grid of sample predictions
├── src/
│   ├── train.py               # trains the model and saves artifacts
│   ├── predict.py             # makes predictions on random samples
│   └── visualize.py           # visualizes predicted vs true labels
├── requirements.txt
├── .gitignore
└── README.md

````

---

## 🚀 Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Praduamn/digitrecognizer.git
   cd digitrecognizer


2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows (PowerShell)
   # source .venv/bin/activate # macOS/Linux
   pip install -r requirements.txt
   ```

---

## 🏋️ Training the Model

Run:

```bash
python src/train.py
```

This will:

* Train the SVM model
* Save the trained model into `artifacts/model.joblib`
* Generate evaluation files:

  * `metrics.txt` → Accuracy + classification report
  * `confusion_matrix.png` → Confusion matrix plot
  * `samples.png` → Grid of predicted vs true digits

---

## 🔎 Making Predictions

Run:

```bash
python src/predict.py --n 10
```

This will load the saved model and predict labels for **10 random test digits**, saving results to:

* `artifacts/sample_predictions.csv`

---

## 👀 Visualizing Predictions

Run:

```bash
python src/visualize.py
```

This will open a **2×5 grid of images** showing:

* The digit image
* Predicted label vs True label

Example title: `P: 4 / T: 4`

---

## 📊 Results

* Model: **SVM (RBF kernel, C=10)**
* Accuracy: \~98–99%
* Confusion matrix and sample predictions are saved in `artifacts/`.

---



