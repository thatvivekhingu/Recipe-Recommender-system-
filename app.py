from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import requests
from dotenv import load_dotenv
import os

load_dotenv()
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


app = Flask(__name__)

try:
    with open('recipe_names.pkl', 'rb') as f:
        recipe_names = pickle.load(f)

    with open('tfid_ingredients.pkl', 'rb') as f:
        tfidf_ingredients = pickle.load(f)

    with open('tfid_diet.pkl', 'rb') as f:
        tfidf_diet = pickle.load(f)

    with open('tfid_course.pkl', 'rb') as f:
        tfidf_course = pickle.load(f)

    with open('tfid_region.pkl', 'rb') as f:
        tfidf_region = pickle.load(f)

    input_features = load_npz('input_features.npz')
except Exception as e:
    print("Error loading models:", e)
    raise
        


def recommend_recipe(ingredients, diet="", course="", region="", top_k=5):
    try:
        ing_vec = tfidf_ingredients.transform([ingredients])
        diet_vec = tfidf_diet.transform([diet])
        course_vec = tfidf_course.transform([course])
        region_vec = tfidf_region.transform([region])

        user_vector = np.hstack([
            ing_vec.toarray(),
            diet_vec.toarray(),
            course_vec.toarray(),
            region_vec.toarray()
        ])

        similarity = cosine_similarity(user_vector, input_features).flatten()
        top_indices = similarity.argsort()[-top_k:][::-1]

        recommendations = []
        for i in top_indices:
            recipe_name = recipe_names[i]
            recipe = {
                'name': recipe_name,
                'similarity_score': round(float(similarity[i]) * 100, 2)
            }
            # Fetch image from Unsplash
            try:
                unsplash_url = (
                    f"https://api.unsplash.com/photos/random"
                    f"?query={recipe_name} indian food&client_id={UNSPLASH_API_KEY}"
                )
                response = requests.get(unsplash_url, timeout=5)
                data = response.json()
                image_url = data.get("urls", {}).get("regular", "")
                recipe["image_url"] = image_url
            except Exception as e:
                print("Image fetch failed:", e)
                recipe["image_url"] = ""
            recommendations.append(recipe)

        return recommendations

    except Exception as e:
        return [{'error': str(e)}]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        ingredients = request.form.get('ingredients', '')
        diet = request.form.get('diet', '')
        course = request.form.get('course', '')
        region = request.form.get('region', '')

        print(f"Received Ingredients: {ingredients}") # <-- ADD THIS

        results = recommend_recipe(ingredients, diet, course, region)
        
        print(f"Results: {results}") # <-- AND THIS

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)



import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_recipes(ingredients, region="India"):
    prompt = f"Suggest 3 unique recipes using these ingredients: {ingredients}. Focus on {region} style variations."
    response = model.generate_content(prompt)
    return response.text
def get_image(query):
    try:
        url = f"https://api.unsplash.com/search/photos?query={query}&client_id={UNSPLASH_API_KEY}"
        res = requests.get(url).json()
        if res.get("results"):
            return res["results"][0]["urls"]["regular"]
        return "/static/default.jpg"  # fallback image
    except:
        return "/static/default.jpg"
