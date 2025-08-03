from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz



app = Flask(__name__)

try:
    with open('models/recipe_names.pkl', 'rb') as f:
        recipe_names = pickle.load(f)

    with open('models/tfid_ingredients.pkl', 'rb') as f:
        tfidf_ingredients = pickle.load(f)

    with open('models/tfid_diet.pkl', 'rb') as f:
        tfidf_diet = pickle.load(f)

    with open('models/tfid_course.pkl', 'rb') as f:
        tfidf_course = pickle.load(f)

    with open('models/tfid_region.pkl', 'rb') as f:
        tfidf_region = pickle.load(f)

    input_features = load_npz('models/input_features.npz')
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
            recipe = {
                'name': recipe_names[i],
                'similarity_score': round(float(similarity[i]) * 100, 2)
            }
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

        results = recommend_recipe(ingredients, diet, course, region)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
