# 🐔 Transfer Learning-Based Classification of Poultry Diseases for Enhanced Health Management

## 📌 Project Overview

Poultry diseases are a major challenge in the poultry farming industry, affecting bird health, productivity, and farm economics. Early identification of diseases is essential to prevent outbreaks and reduce losses.

This project develops an AI-powered poultry disease classification system using **Deep Learning and Transfer Learning**. It analyzes poultry images and predicts whether the sample belongs to one of four categories: Coccidiosis, Salmonella, Newcastle Disease, or Healthy.

The trained model is integrated into a web application, allowing users to upload images and receive automated predictions along with confidence scores. The system also aims to prevent invalid or unrelated images from being processed as poultry disease samples.

## 🎯 Project Objectives

- Automate poultry disease classification using Deep Learning.
- Apply Transfer Learning to improve image classification.
- Identify common poultry diseases at an early stage.
- Provide prediction results with confidence scores.
- Develop a simple and user-friendly web interface.
- Support better poultry health monitoring and management.

## 🚀 Key Features

- **AI-Based Disease Classification:** Identifies poultry health conditions from uploaded images.
- **Transfer Learning:** Utilizes pre-trained deep learning models for image classification.
- **Multi-Class Prediction:** Recognizes four different categories.
- **Confidence Score:** Displays the model's prediction confidence.
- **Image Validation:** Helps prevent unrelated images from being treated as valid poultry samples.
- **Web-Based Interface:** Provides an accessible interface for uploading images and viewing predictions.
- **Automated Results:** Generates disease classification results without manual image inspection.

## 🦠 Disease Categories

The system classifies images into the following four categories:

| Category | Description |
|---|---|
| Healthy | Indicates a healthy poultry sample |
| Coccidiosis | Identifies signs associated with Coccidiosis |
| Salmonella | Identifies signs associated with Salmonella |
| Newcastle Disease | Identifies signs associated with Newcastle Disease |

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** Convolutional Neural Networks (CNN)
- **Transfer Learning:** Pre-trained deep learning architecture
- **Framework:** TensorFlow / Keras
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Image Processing:** Image preprocessing and normalization
- **Version Control:** Git and GitHub

## ⚙️ System Workflow

1. User opens the poultry disease classification web application.
2. User uploads a poultry image.
3. The uploaded image undergoes preprocessing and validation.
4. The trained transfer learning model analyzes the image.
5. The model predicts one of the four disease categories.
6. The application displays the predicted category and confidence score.

## 🔄 Project Architecture

```text
          User
            |
            v
      Web Application
            |
            v
       Image Upload
            |
            v
    Image Preprocessing
            |
            v
      Image Validation
            |
            v
   Transfer Learning Model
            |
            v
   Disease Classification
            |
            v
     Confidence Score
            |
            v
     Prediction Results
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Tejasri4477/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management.git
```

### 2. Navigate to the Project Directory

```bash
cd Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open in Browser

```text
http://127.0.0.1:5000/
```

*Note: Ensure the trained model and required dataset files are available in their expected project directories before running the application.*

## 📊 Expected Output

The application provides:

- Uploaded image preview.
- Predicted poultry disease category.
- Prediction confidence score.
- Classification result through the web interface.

## 🌱 Future Enhancements

- Improve classification accuracy using larger and more diverse datasets.
- Explore additional transfer learning architectures.
- Enhance image validation and disease prediction reliability.
- Develop a mobile-friendly application.
- Integrate cloud-based model deployment.
- Provide additional poultry disease information and preventive recommendations.

## ⚠️ Disclaimer

This project is an AI-based research and learning prototype. Its predictions are intended to support preliminary poultry health assessment and should not replace professional veterinary diagnosis.

## 👩‍💻 Author

**Somavarapu Tejasri**

- GitHub: [Tejasri4477](https://github.com/Tejasri4477)

---

⭐ If you find this project useful, consider giving the repository a star!
