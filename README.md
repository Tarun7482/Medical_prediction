# MediPredict

MediPredict is an AI-powered web application designed to help users gain insights into potential health conditions based on their symptoms. Built with Flask, Python, and modern web technologies, MediPredict aims to make medical knowledge more accessible and empower users to take charge of their health.

## Features

- **Symptom-based Prediction:** Enter your symptoms and receive possible health condition predictions.
- **AI/ML Powered:** Utilizes machine learning models trained on medical datasets for accurate predictions.
- **Modern UI:** Responsive and user-friendly interface built with Tailwind CSS.
- **Developer Info:** Learn about the developer and the project journey on the About page.

## Technologies Used

- **Flask:** Python web framework for backend and routing.
- **Python:** Core programming language for backend logic and machine learning.
- **Machine Learning:** Scikit-learn and related libraries for model training and prediction.
- **Tailwind CSS:** For modern, responsive, and clean UI design.

## Project Structure

```
medical_pre/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── models/                 # ML models and encoders
│   ├── model.pkl
│   ├── label_encoder.pkl
│   └── symptom_vocab.pkl
├── static/                 # Static files (images, CSS, JS)
│   └── image.jpg
├── templates/              # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   └── predict.html
```

## Getting Started

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd medical_pre
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```
4. **Open in browser:**
   Visit `http://127.0.0.1:5000` in your web browser.

## Usage
- Go to the home page and enter your symptoms.
- View the predicted health conditions and suggested next steps.
- Visit the About page to learn more about the developer and the technologies used.

## Developer
**Tarun Kumar**  
[GitHub](https://github.com/Tarun7482) | [LinkedIn](https://www.linkedin.com/in/tarun-kumar-65a8232a6/)

## Project Journey
- **Project Inception:** Started with the idea to make medical knowledge more accessible.
- **Data Collection:** Gathered and cleaned medical datasets for training.
- **Model Development:** Built and trained the prediction algorithms.
- **Launch:** Released MediPredict to help people understand their symptoms.

## License
This project is for educational and demonstration purposes. For any commercial or clinical use, consult a medical professional and comply with relevant regulations.

---

*Empowering users with accessible medical insights through technology.*
