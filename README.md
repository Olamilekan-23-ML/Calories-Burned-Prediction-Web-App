# 🔥 Calories Burned Prediction Web App
This is a Machine Learning Web Application that predicts calories burned during exercise based on user activity, demographic,and
physiological input data. It uses a trained Regression model and a Streamlit web interface for user-friendly interaction.

---
# 🚀 Project Overview
• **Model Type**: Supervised Machine Learning (Regression)

• **Web Framework**: Streamlit

• **Core Libraries**: Scikit-learn, Pandas, NumPy, Pickle

• **Language**: Python

_The user provides details such as exercise type, duration, heart rate, body temperature, and other factors through a web form. 
The app processes the inputs and predicts the approximate number of calories burned, offering personalized fitness insights._

---
# 🧠 How It Works
1. A pre-trained Regression model (trained on exercise and calorie expenditure datasets) is loaded using `pickle`.
2. The user inputs their activity information via the interactive Streamlit form.
3. The app processes and prepares the input data (handling categorical variables like Gender, Exercise Type).
4. The model makes a prediction, and the result is displayed instantly with actionable fitness recommendations.

---
# 💻 How to Run the App Locally
## 1️⃣ Clone the Repository
### Open your terminal (Command Prompt, PowerShell, or Terminal) and run the following commands:
``git clone https://github.com/Olamilekan-23-ML/Calories-Burned-Prediction-Web-App.git
cd Calories-Burned-Prediction-Web-App``
## 2️⃣ Install Dependencies
### Ensure you have Python installed, then run:
``pip install -r requirements.txt``
## 3️⃣ Run the Streamlit App
### Start the application with the following command:
``streamlit run Calories.py``
_Then open the URL shown in your terminal (usually http://localhost:8501) in your web browser._

---
# 📂 Project Structure
| File | Description |
| :--- | :--- |
| `Calories.py` | The main Streamlit app script that runs the web interface. |
| `mymodel.pkl` | The serialized, trained machine learning model. |
| `Calories_Prediction.py` | The Python file containing the complete data analysis and model training code. |
| `calories.csv` |The primary dataset used for training the model. |
| `exercise.csv` |Supplementary exercise dataset. |
| `requirements.txt` | List of Python dependencies required to run the app. |
| `README.md` | This file. |

---
🧰 Technologies Used
• Python

• Streamlit

• Scikit-learn

• NumPy & Pandas

• Pickle

----
# ⚠️ Important Disclaimer
_This tool is for educational and informational purposes only. It is not a substitute for professional medical, fitness, or 
nutritional advice. Always consult with qualified healthcare or fitness professionals before starting any exercise program or 
making changes to your health regimen._

# 👤 Author
*_OLAMILEKAN_*

*_GitHub: @Olamilekan-23-ML_*
