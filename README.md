# ❤️ Heart Disease Prediction App

A simple **machine learning web app** that predicts the risk of heart disease based on user health inputs.  
Built with **Python, scikit-learn, and Streamlit**.

---

## 🚀 Features
- User-friendly Streamlit interface
- Predicts “Low Risk” or “High Risk” of heart disease
- Uses a KNN model trained on heart disease dataset
- Live deployment on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 🛠️ Tech Stack
- Python 3
- scikit-learn
- pandas, numpy
- Streamlit

---

## 📂 Project Structure
heart-disease-predictor/
│
├── app.py # Streamlit app
├── KNN_heart.pkl # Trained model
├── scaler.pkl # Scaler used for preprocessing
├── columns.pkl # Expected columns for model
├── requirements.txt # Project dependencies
└── README.md

## ▶️ Run Locally
1. Clone this repo:
   git clone https://github.com/YOURUSERNAME/heart-disease-predictor.git
   cd heart-disease-predictor
2. Create & activate virtual environment:
   python -m venv venv
   .\venv\Scripts\activate      # (Windows PowerShell)
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   streamlit run app.py

🌐 Deployment
Deployed on Streamlit Community Cloud:
👉 https://heartdiseasepredictor-nsax7vd3xnefvhiy2jz2e5.streamlit.app/

⚠️ Disclaimer
This app is for educational/demo purposes only and is not a substitute for professional medical advice or diagnosis.
