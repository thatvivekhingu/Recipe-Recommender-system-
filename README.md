<div align="center">

# 🍳 Smart Recipe Recommender System

### *Content-Based AI Recommender Engine for Culinary & Meal Planning*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-TF--IDF-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![HTML5/CSS3](https://img.shields.io/badge/Frontend-HTML5%2FCSS3-E34F26?style=for-the-badge)

</div>

---

## 📌 Overview

The **Recipe Recommender System** is an intelligent machine learning web application that suggests personalized culinary recipes based on user inputs, available ingredients, dietary preferences, course types, and regional cuisines. Utilizing **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization and **Cosine Similarity**, the engine calculates mathematical similarity scores across multi-attribute feature vectors to recommend the top matching recipes.

---

## 🧮 Mathematical Formulation & ML Pipeline

```
Feature Vector Construction:
  V = [ TF-IDF(Ingredients)  ||  TF-IDF(Diet)  ||  TF-IDF(Course)  ||  TF-IDF(Region) ]

Cosine Similarity Metric:
  Similarity(A, B) = (A · B) / (||A|| × ||B||)
```

1. **Text Preprocessing**: Normalization, tokenization, and stop-word removal on ingredients and meal attributes.
2. **Multi-Vector TF-IDF Encoding**: Feature matrices generated and saved as compressed NumPy sparse arrays (`.npz`).
3. **Similarity Engine**: Fast matrix dot-products to output top-K recommended recipes in under 50ms.

---

## ✨ Features

- 🥗 **Ingredient-Based Search**: Input available pantry ingredients to find instant cooking ideas.
- 🥑 **Dietary Filtering**: Filter by Vegetarian, Non-Vegetarian, Vegan, Gluten-Free, and Keto diets.
- 🥘 **Regional Cuisine Filter**: Explore traditional Indian, Continental, Asian, and Mediterranean recipes.
- ⚡ **Instant Recommendations**: High-speed Flask backend with pre-computed vector models.

---

## 📂 Project Structure

```
Recipe-Recommender-system-/
├── app.py                      # Flask Application Server
├── my_andom_util.py           # Helper Functions & Preprocessing Utilities
├── input_features.npz         # Vectorized Sparse Feature Matrix
├── recipe_names.pkl           # Pickled Recipe Index Dataset
├── tfid_ingredients.pkl       # TF-IDF Vectorizer for Ingredients
├── tfid_course.pkl            # TF-IDF Vectorizer for Courses
├── tfid_diet.pkl              # TF-IDF Vectorizer for Diets
├── tfid_region.pkl            # TF-IDF Vectorizer for Regions
├── templates/
│   └── index.html             # Web Application Frontend UI
└── README.md                  # Project Documentation
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/thatvivekhingu/Recipe-Recommender-system-.git
cd Recipe-Recommender-system-

# Install required dependencies
pip install flask numpy scikit-learn

# Run the Flask app
python app.py
```

Access the application in your browser at `http://127.0.0.1:5000`.

---

## 📄 License

This project is open-source and available under the **MIT License**.
