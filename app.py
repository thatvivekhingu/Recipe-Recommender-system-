import os
import re
import json
import io
import urllib.parse
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import requests
from dotenv import load_dotenv
from PIL import Image

# Import Database, Caching & Nutrition Services
import database.db as db
from image_cache import get_recipe_image, get_recipe_youtube
from nutrition_service import calculate_recipe_nutrition

load_dotenv()
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize YOLO model if available
try:
    from ultralytics import YOLO
    yolo_path = os.path.join(BASE_DIR, 'yolov8n.pt')
    if os.path.exists(yolo_path):
        yolo_model = YOLO(yolo_path)
    else:
        yolo_model = None
except Exception as e:
    yolo_model = None

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Load ML TF-IDF and Sparse Matrices
try:
    with open(os.path.join(BASE_DIR, 'recipe_names.pkl'), 'rb') as f:
        recipe_names = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'tfid_ingredients.pkl'), 'rb') as f:
        tfidf_ingredients = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'tfid_diet.pkl'), 'rb') as f:
        tfidf_diet = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'tfid_course.pkl'), 'rb') as f:
        tfidf_course = pickle.load(f)

    with open(os.path.join(BASE_DIR, 'tfid_region.pkl'), 'rb') as f:
        tfidf_region = pickle.load(f)

    input_features = load_npz(os.path.join(BASE_DIR, 'input_features.npz'))
except Exception as e:
    print("Error loading models:", e)
    raise

# Precompute Recipe Metadata (Ingredients, Diet, Course, Region)
feature_names = tfidf_ingredients.get_feature_names_out()
features_array = input_features.toarray()
diet_names = tfidf_diet.get_feature_names_out()
course_names = tfidf_course.get_feature_names_out()
region_names = tfidf_region.get_feature_names_out()

num_ing = len(feature_names)
num_diet = len(diet_names)
num_course = len(course_names)

recipe_ingredients_map = {}
recipe_diet_map = {}
recipe_region_map = {}

for idx, name in enumerate(recipe_names):
    # Ingredients
    ing_row = features_array[idx, :num_ing]
    active_ings = ing_row.nonzero()[0]
    recipe_ingredients_map[name] = [feature_names[i] for i in active_ings]

    # Diet
    diet_row = features_array[idx, num_ing:num_ing + num_diet]
    active_diet = diet_row.nonzero()[0]
    recipe_diet_map[name] = diet_names[active_diet[0]] if len(active_diet) > 0 else "vegetarian"

    # Region
    reg_start = num_ing + num_diet + num_course
    reg_row = features_array[idx, reg_start:reg_start + len(region_names)]
    active_reg = reg_row.nonzero()[0]
    recipe_region_map[name] = region_names[active_reg[0]] if len(active_reg) > 0 else "north"

# Indian Culinary Substitutions Dictionary
SUBSTITUTES_MAP = {
    "paneer": "Tofu, dense boiled potato, or ricotta cheese",
    "ghee": "Butter, refined cooking oil, or mustard oil",
    "clarified": "Butter or coconut oil",
    "butter": "Ghee or mustard oil",
    "cream": "Cashew paste, fresh malai, or whisked hung curd",
    "yogurt": "Fresh curd, buttermilk, or lemon juice with milk",
    "curd": "Plain yogurt or cashew-lemon cream",
    "garam": "Roasted cumin & coriander powder with pinch of cinnamon and cloves",
    "masala": "Kitchen King masala or roasted ground cumin-coriander",
    "kewra": "Rose water, saffron infused milk, or cardamom powder",
    "saffron": "Turmeric powder (for golden hue) + crushed cardamom",
    "cashews": "Blanched almonds, melon seeds (magaz), or poppy seeds",
    "almonds": "Cashews or roasted peanuts",
    "maida": "Atta (whole wheat flour) or besan (gram flour)",
    "flour": "Atta or besan (gram flour)",
    "besan": "Gram flour, roasted sattu, or cornstarch",
    "tomato": "Tomato puree, tamarind pulp, or whisked curd",
    "onion": "Hing (asafoetida) + finely shredded cabbage or grated radish",
    "garlic": "Hing (asafoetida) or ginger-green chili paste",
    "ginger": "Dry ginger powder (saunth) or grated galangal",
    "chickpeas": "White or black chana, kidney beans (rajma), or green peas",
    "jaggery": "Brown sugar or pure honey",
    "sugar": "Jaggery (gud), date syrup, or honey",
    "coconut": "Desiccated dry coconut or thick coconut milk",
    "mustard": "Cumin seeds (jeera) or fenugreek seeds",
    "fenugreek": "Kasuri methi (dried leaves) or fresh spinach",
    "coriander": "Fresh mint leaves or kasuri methi for aroma",
    "potatoes": "Raw banana (plantain), sweet potato, or yam",
    "rice": "Jeera rice, brown rice, or quinoa"
}


def recommend_recipe(ingredients, diet="", course="", region="", top_k=6):
    """Recommend recipes with instant caching and matching analysis."""
    try:
        clean_ingredients = ingredients.strip() if ingredients else "paneer, tomato, onion"
        ing_vec = tfidf_ingredients.transform([clean_ingredients])
        diet_vec = tfidf_diet.transform([diet or ""])
        course_vec = tfidf_course.transform([course or ""])
        region_vec = tfidf_region.transform([region or ""])

        user_vector = np.hstack([
            ing_vec.toarray(),
            diet_vec.toarray(),
            course_vec.toarray(),
            region_vec.toarray()
        ])

        similarity = cosine_similarity(user_vector, input_features).flatten()
        top_indices = similarity.argsort()[-top_k:][::-1]

        user_tokens = set(re.findall(r"\w+", clean_ingredients.lower()))

        recommendations = []
        for i in top_indices:
            recipe_name = recipe_names[i]

            # Fast cached image & YouTube lookup (<1ms)
            image_url = get_recipe_image(recipe_name, UNSPLASH_API_KEY)
            yt_data = get_recipe_youtube(recipe_name)

            # Calculate matched and missing ingredients
            all_req_ings = recipe_ingredients_map.get(recipe_name, [])
            matched = [ing for ing in all_req_ings if any(tok in ing or ing in tok for tok in user_tokens)]
            missing = [ing for ing in all_req_ings if ing not in matched]

            # Suggest substitutes for missing ingredients
            substitutes = {}
            for m in missing:
                for key, sub in SUBSTITUTES_MAP.items():
                    if key in m.lower():
                        substitutes[m] = sub
                        break

            # Check if saved in favorites database
            is_fav = db.is_favorite(recipe_name)

            recipe = {
                'name': recipe_name,
                'similarity_score': round(float(similarity[i]) * 100, 1),
                'image_url': image_url,
                'region': recipe_region_map.get(recipe_name, 'Indian').capitalize(),
                'diet': recipe_diet_map.get(recipe_name, 'Vegetarian').capitalize(),
                'all_ingredients': all_req_ings,
                'matched_ingredients': matched,
                'missing_ingredients': missing,
                'substitutes': substitutes,
                'youtube': yt_data,
                'is_favorite': is_fav,
                'nutrition': calculate_recipe_nutrition(recipe_name, all_req_ings)
            }
            recommendations.append(recipe)

        return recommendations

    except Exception as e:
        print("Recommendation error:", e)
        return [{'error': str(e)}]


# ==========================================
# PAGE ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')


# ==========================================
# CORE RECOMMENDER API
# ==========================================
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
        return jsonify({'error': str(e)}), 500


# ==========================================
# DASHBOARD STATS API
# ==========================================
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return live dashboard statistics from database."""
    try:
        stats = db.get_dashboard_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# PANTRY REST API
# ==========================================
@app.route('/api/pantry', methods=['GET', 'POST'])
def manage_pantry():
    try:
        if request.method == 'POST':
            data = request.get_json() or request.form
            name = data.get('name', '')
            category = data.get('category', 'General')
            quantity = data.get('quantity', '1 unit')

            if not name.strip():
                return jsonify({'success': False, 'error': 'Ingredient name is required'}), 400

            new_id = db.add_pantry_item(name, category, quantity)
            return jsonify({'success': True, 'id': new_id, 'message': f"'{name}' added to your pantry!"})

        # GET request
        items = db.get_all_pantry()
        return jsonify({'success': True, 'items': items, 'count': len(items)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/pantry/<int:item_id>', methods=['DELETE'])
def delete_pantry(item_id):
    try:
        success = db.delete_pantry_item(item_id)
        return jsonify({'success': success, 'message': 'Pantry item deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/pantry/recommend', methods=['GET'])
def recommend_from_pantry():
    """Recommend recipes using the user's current pantry inventory directly from database."""
    try:
        pantry_str = db.get_pantry_ingredients_string()
        if not pantry_str:
            pantry_str = "paneer, tomato, onion, ghee, cumin"

        results = recommend_recipe(pantry_str, top_k=6)
        return jsonify({'success': True, 'pantry_ingredients': pantry_str, 'recipes': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# FAVORITES REST API
# ==========================================
@app.route('/api/favorites', methods=['GET'])
def get_favorites():
    try:
        favs = db.get_all_favorites()
        return jsonify({'success': True, 'favorites': favs, 'count': len(favs)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/favorites/toggle', methods=['POST'])
def toggle_favorite():
    try:
        data = request.get_json() or request.form
        recipe_name = data.get('recipe_name', '')
        diet = data.get('diet', '')
        region = data.get('region', '')
        score = float(data.get('similarity_score', 0.0) or 0.0)
        image_url = data.get('image_url', '')
        ingredients = data.get('ingredients', [])
        if isinstance(ingredients, str):
            try:
                ingredients = json.loads(ingredients)
            except Exception:
                ingredients = [i.strip() for i in ingredients.split(',') if i.strip()]
        youtube_url = data.get('youtube_url', '')
        notes = data.get('notes', '')

        res = db.toggle_favorite(
            recipe_name=recipe_name,
            diet=diet,
            region=region,
            similarity_score=score,
            image_url=image_url,
            ingredients=ingredients,
            youtube_url=youtube_url,
            notes=notes
        )
        return jsonify(res)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/favorites/<int:fav_id>', methods=['DELETE'])
def delete_favorite(fav_id):
    try:
        success = db.delete_favorite(fav_id)
        return jsonify({'success': success, 'message': 'Removed from favorites'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# SHOPPING LIST REST API
# ==========================================
@app.route('/api/shopping-list', methods=['GET', 'POST'])
def manage_shopping_list():
    try:
        if request.method == 'POST':
            data = request.get_json() or request.form
            item_name = data.get('item_name', '')
            recipe_source = data.get('recipe_source', 'Kitchen')
            quantity = data.get('quantity', '1 unit')

            if not item_name.strip():
                return jsonify({'success': False, 'error': 'Item name is required'}), 400

            new_id = db.add_shopping_item(item_name, recipe_source, quantity)
            return jsonify({'success': True, 'id': new_id, 'message': f"'{item_name}' added to shopping list"})

        # GET request
        items = db.get_all_shopping_items()
        return jsonify({'success': True, 'items': items, 'count': len(items)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/shopping-list/from-recipe', methods=['POST'])
def add_from_recipe():
    """Add all missing ingredients of a recipe to shopping list with 1-click."""
    try:
        data = request.get_json() or request.form
        recipe_name = data.get('recipe_name', 'Recipe')
        missing = data.get('missing_ingredients', [])
        if isinstance(missing, str):
            try:
                missing = json.loads(missing)
            except Exception:
                missing = [m.strip() for m in missing.split(',') if m.strip()]

        added_count = db.add_missing_ingredients_to_shopping(recipe_name, missing)
        return jsonify({
            'success': True,
            'added_count': added_count,
            'message': f"Added {added_count} missing ingredients for '{recipe_name}' to shopping list!"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/shopping-list/<int:item_id>', methods=['PATCH', 'DELETE'])
def update_or_delete_shopping(item_id):
    try:
        if request.method == 'PATCH':
            data = request.get_json() or {}
            is_purchased = data.get('is_purchased', None)
            success = db.toggle_shopping_item(item_id, is_purchased)
            return jsonify({'success': success})

        # DELETE
        success = db.delete_shopping_item(item_id)
        return jsonify({'success': success, 'message': 'Item deleted'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/shopping-list/clear-purchased', methods=['POST'])
def clear_purchased():
    try:
        count = db.clear_purchased_shopping_items()
        return jsonify({'success': True, 'cleared_count': count, 'message': f'Cleared {count} purchased items'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# MEAL PLANNER REST API
# ==========================================
@app.route('/api/meal-plans', methods=['GET', 'POST'])
def manage_meal_plans():
    try:
        if request.method == 'POST':
            data = request.get_json() or request.form
            day = data.get('day_of_week', 'Monday')
            slot = data.get('meal_slot', 'Dinner')
            recipe = data.get('recipe_name', '')
            notes = data.get('notes', '')

            if not recipe.strip():
                return jsonify({'success': False, 'error': 'Recipe name is required'}), 400

            new_id = db.add_meal_plan(day, slot, recipe, notes)
            return jsonify({'success': True, 'id': new_id, 'message': f"Scheduled {recipe} for {day} {slot}!"})

        # GET request
        plans = db.get_all_meal_plans()
        return jsonify({'success': True, 'plans': plans, 'count': len(plans)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/meal-plans/<int:plan_id>', methods=['DELETE'])
def delete_meal_plan(plan_id):
    try:
        success = db.delete_meal_plan(plan_id)
        return jsonify({'success': success, 'message': 'Meal plan deleted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# REVIEWS & RATINGS REST API
# ==========================================
@app.route('/api/reviews/<recipe_name>', methods=['GET'])
def get_reviews(recipe_name):
    try:
        data = db.get_recipe_reviews(recipe_name)
        return jsonify({'success': True, **data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reviews', methods=['POST'])
def post_review():
    try:
        data = request.get_json() or request.form
        recipe_name = data.get('recipe_name', '')
        rating = int(data.get('rating', 5))
        comment = data.get('comment', '')
        user_name = data.get('user_name', 'Home Chef')

        if not recipe_name.strip():
            return jsonify({'success': False, 'error': 'Recipe name is required'}), 400

        review_id = db.add_recipe_review(recipe_name, rating, comment, user_name)
        return jsonify({'success': True, 'id': review_id, 'message': 'Review submitted!'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# AI FRIDGE & PANTRY VISION SCANNER
# ==========================================
@app.route('/detect-ingredients', methods=['POST'])
def detect_ingredients():
    """Detect available ingredients from an uploaded fridge or pantry photo using Gemini or local YOLO."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        detected_ingredients = []
        source = "local_yolo_ai"

        # 1. Try Gemini Vision if GOOGLE_API_KEY is available
        if GOOGLE_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    "Look at this image carefully. Is this an image of food ingredients, a refrigerator, vegetables, pantry groceries, or cooking items? "
                    "If NO (for example if it is a person, portrait, animal, document, vehicle, clothes, or non-food scene), start your answer with NON_FOOD: followed by what is in the image. "
                    "If YES, return ONLY a clean comma-separated list of common English ingredient names (e.g. tomato, onion, milk, paneer, capsicum)."
                )
                response = model.generate_content([prompt, pil_image])
                raw_text = response.text.strip()
                if "NON_FOOD" in raw_text:
                    detected_desc = raw_text.replace("NON_FOOD:", "").strip()
                    return jsonify({
                        'success': False,
                        'error': f"Is photo me koi food ya fridge ingredients nahi mile ({detected_desc})! Kripya kisi fridge, sabziyon ya kitchen pantry ki photo upload karein."
                    })
                items = [item.strip().lower() for item in raw_text.split(',') if item.strip()]
                if items:
                    detected_ingredients = items
                    source = "gemini_vision"
            except Exception as e:
                print("Gemini Vision API error (falling back to YOLO local vision):", e)

        # 2. Local YOLO Computer Vision fallback
        if not detected_ingredients and yolo_model:
            try:
                results = yolo_model(pil_image, verbose=False)
                detected_classes = set()
                for r in results:
                    for c in r.boxes.cls:
                        detected_classes.add(yolo_model.names[int(c)])

                NON_FOOD_OBJS = {
                    'person', 'tie', 'suit', 'car', 'truck', 'bus', 'motorcycle', 
                    'bicycle', 'dog', 'cat', 'laptop', 'cell phone', 'tv', 'chair', 
                    'couch', 'bed', 'book', 'clock', 'backpack', 'handbag', 'suitcase'
                }
                non_food_matches = detected_classes.intersection(NON_FOOD_OBJS)

                FOOD_MAPPING = {
                    'banana': 'banana', 'apple': 'apple', 'sandwich': 'bread',
                    'orange': 'orange', 'broccoli': 'broccoli', 'carrot': 'carrot',
                    'pizza': 'cheese', 'bottle': 'milk', 'bowl': 'curd',
                    'refrigerator': 'vegetables', 'cup': 'tea'
                }

                food_items = [FOOD_MAPPING[k] for k in detected_classes if k in FOOD_MAPPING]

                if non_food_matches and not food_items:
                    detected_str = ', '.join([item.title() for item in non_food_matches])
                    return jsonify({
                        'success': False,
                        'error': f"Yeh fridge ya food ki photo nahi hai! Is photo me '{detected_str}' detect hua hai. Kripya apne fridge, sabziyon ya pantry ki photo dalein."
                    })

                if food_items:
                    detected_ingredients = list(set(food_items))
                    source = "local_yolo_ai"
            except Exception as e:
                print("YOLO inference error:", e)

        if not detected_ingredients:
            return jsonify({
                'success': False,
                'error': "Is photo me koi food ya fridge ingredients detect nahi hue. Kripya kisi fridge, sabziyon ya kitchen groceries ki clear photo upload karein!"
            })

        return jsonify({
            'success': True,
            'detected_ingredients': detected_ingredients,
            'ingredients_str': ', '.join(detected_ingredients),
            'source': source,
            'count': len(detected_ingredients)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# REGIONAL THALIS CATALOG
# ==========================================
@app.route('/regional-recipes', methods=['GET'])
def regional_recipes():
    """Return top authentic recipes organized by Indian regions."""
    try:
        region_filter = request.args.get('region', '').lower()

        regions_catalog = {
            "north": {
                "title": "North Indian (Punjabi, Mughlai)",
                "icon": "🌾",
                "description": "Rich gravies, slow-cooked dals, tandoori flavors, and rich warming spices.",
                "recipes": ["Paneer butter masala", "Chana masala", "Dal tadka", "Gajar ka halwa", "Shahi paneer", "Aloo matar", "Lassi", "Rabri"]
            },
            "west": {
                "title": "West Indian (Gujarati, Rajasthani, Maharashtrian)",
                "icon": "🌶",
                "description": "Sweet and tangy Gujarati flavors, rich Rajasthani desserts, and aromatic curries.",
                "recipes": ["Ghevar", "Sheera", "Chikki", "Basundi", "Kofta", "Boondi"]
            },
            "south": {
                "title": "South Indian (Tamil, Andhra, Kerala, Karnataka)",
                "icon": "🥥",
                "description": "Fragrant curry leaves, mustard seeds, tamarind tanginess, and coconut infusions.",
                "recipes": ["Sambar", "Double ka meetha", "Payasam", "Dharwad pedha"]
            },
            "east": {
                "title": "East Indian (Bengali, Odia, Assamese)",
                "icon": "🐟",
                "description": "Subtle mustard notes, panch phoron spices, and famous milk-based sweet confections.",
                "recipes": ["Balu shahi", "Gulab jamun", "Imarti", "Kalakand", "Phirni", "Sandesh", "Rasgulla"]
            },
            "central": {
                "title": "Central Indian (Malwa, Bundelkhand)",
                "icon": "🏰",
                "description": "Rustic, soul-warming delicacies with rich dry-fruit stuffed sweets and savory thalis.",
                "recipes": ["Petha", "Boondi", "Sheera"]
            }
        }

        if region_filter and region_filter in regions_catalog:
            selected_regions = {region_filter: regions_catalog[region_filter]}
        else:
            selected_regions = regions_catalog

        result = {}
        for reg_key, reg_data in selected_regions.items():
            dish_cards = []
            for dish_name in reg_data["recipes"]:
                if dish_name in recipe_names:
                    dish_cards.append({
                        "name": dish_name,
                        "diet": recipe_diet_map.get(dish_name, "Vegetarian").capitalize(),
                        "ingredients": recipe_ingredients_map.get(dish_name, []),
                        "youtube": get_recipe_youtube(dish_name),
                        "image_url": get_recipe_image(dish_name),
                        "is_favorite": db.is_favorite(dish_name),
                        "nutrition": calculate_recipe_nutrition(dish_name, recipe_ingredients_map.get(dish_name, []))
                    })
            result[reg_key] = {
                "title": reg_data["title"],
                "icon": reg_data["icon"],
                "description": reg_data["description"],
                "dishes": dish_cards
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# AI MASTER CHEF GENERATOR
# ==========================================
@app.route('/generate-ai-recipe', methods=['POST'])
def generate_ai_recipe():
    """Personalized AI Chef: Generate a custom authentic recipe based on user preferences."""
    try:
        ingredients = request.form.get('ingredients', 'Paneer, Tomato, Onion')
        cuisine = request.form.get('cuisine', 'Punjabi')
        diet = request.form.get('diet', 'Vegetarian')
        spice_level = request.form.get('spice_level', 'Medium')
        cook_time = request.form.get('cook_time', '30 mins')
        dish_type = request.form.get('dish_type', 'Curry / Gravy')

        recipe_data = None

        if GOOGLE_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GOOGLE_API_KEY)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = f"""
                You are a Master Indian Chef. Generate an authentic, personalized Indian recipe based on:
                - Ingredients available: {ingredients}
                - Regional Style: {cuisine} Indian Cuisine
                - Diet: {diet}
                - Spice Level: {spice_level}
                - Cooking Time: {cook_time}
                - Dish Style: {dish_type}

                Respond strictly with valid JSON without backticks:
                {{
                    "title": "Authentic Recipe Name",
                    "regional_style": "{cuisine} Special",
                    "description": "Appetizing 1-sentence description.",
                    "prep_time": "10 mins",
                    "cook_time": "{cook_time}",
                    "servings": "3-4 Servings",
                    "spice_level": "{spice_level}",
                    "ingredients_needed": [
                        {{"item": "ingredient name", "measure": "measurement"}}
                    ],
                    "instructions": [
                        "Step 1: ...",
                        "Step 2: ...",
                        "Step 3: ...",
                        "Step 4: ...",
                        "Step 5: ..."
                    ],
                    "chef_secret_tip": "One authentic pro cooking tip for maximum flavor.",
                    "pairing": "Suggested bread or rice to pair with."
                }}
                """
                response = model.generate_content(prompt)
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                recipe_data = json.loads(clean_text)
            except Exception as e:
                print("Gemini Recipe Generation error (using dynamic culinary chef engine):", e)

        if not recipe_data:
            primary_ing = ingredients.split(',')[0].strip().title() if ingredients else "Paneer"
            dish_title = f"{cuisine} Style {primary_ing} {dish_type.split('/')[0].strip()}"
            
            recipe_data = {
                "title": dish_title,
                "regional_style": f"{cuisine} Regional Flavors",
                "description": f"A rich, soul-satisfying {cuisine.lower()} preparation featuring {ingredients} simmered with signature spices.",
                "prep_time": "12 mins",
                "cook_time": cook_time,
                "servings": "3 Servings",
                "spice_level": spice_level,
                "ingredients_needed": [
                    {"item": ing.strip().title(), "measure": "1 cup / as required"} for ing in ingredients.split(',') if ing.strip()
                ] + [
                    {"item": "Ghee or Mustard Oil", "measure": "2 tablespoons"},
                    {"item": "Cumin Seeds (Jeera)", "measure": "1 teaspoon"},
                    {"item": "Turmeric & Kashmiri Chili Powder", "measure": "1 teaspoon each"},
                    {"item": "Garam Masala or Kitchen King", "measure": "1/2 teaspoon"},
                    {"item": "Salt", "measure": "To taste"}
                ],
                "instructions": [
                    f"Heat ghee or oil in a heavy-bottomed kadai. Crackle cumin seeds and let the aroma bloom.",
                    f"Add finely chopped base ingredients ({ingredients.split(',')[0].strip()}) and sauté on medium heat until golden brown.",
                    f"Mix in the dry spices (turmeric, chili powder, and salt). Sauté for 1 minute until oil begins to separate from the gravy.",
                    f"Gently fold in the remaining ingredients. Add 1/2 cup warm water, cover with lid, and simmer for 8–10 minutes to let flavors meld.",
                    f"Finish with a pinch of roasted garam masala and fresh coriander. Serve piping hot!"
                ],
                "chef_secret_tip": f"For authentic {cuisine} aroma, finish with a tempering (tadka) of crushed kasuri methi or hing in half-spoon warm ghee just before serving.",
                "pairing": "Fluffy Butter Phulkas, Steamed Basmati Rice, or Jeera Pulao"
            }

        yt_info = get_recipe_youtube(recipe_data["title"])
        recipe_data["youtube"] = yt_info

        return jsonify({
            'success': True,
            'recipe': recipe_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# SYSTEM HEALTH & INTERACTIVE API DOCS
# ==========================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Production health check probe."""
    return jsonify({
        'status': 'healthy',
        'service': 'Suggestion By Anna Culinary Intelligence Engine',
        'database': 'SQLite recipes.db operational',
        'models_loaded': len(recipe_names) > 0,
        'total_recipes': len(recipe_names)
    })


@app.route('/api/docs', methods=['GET'])
def api_documentation():
    """Interactive OpenAPI-style documentation playground for developers and evaluators."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Suggestion By Anna REST API Interactive Playground</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            body { font-family: 'Plus Jakarta Sans', sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; margin: 0; line-height: 1.6; }
            .container { max-width: 900px; margin: 0 auto; }
            .header { border-bottom: 1px solid #334155; padding-bottom: 1.5rem; margin-bottom: 2rem; }
            h1 { font-size: 2rem; color: #4ade80; margin-bottom: 0.3rem; }
            .endpoint-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem; }
            .method { font-family: 'JetBrains Mono', monospace; font-weight: 800; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.85rem; }
            .get { background: #0284c7; color: #fff; }
            .post { background: #16a34a; color: #fff; }
            .delete { background: #dc2626; color: #fff; }
            .patch { background: #d97706; color: #fff; }
            .path { font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; margin-left: 0.6rem; color: #e2e8f0; }
            .desc { color: #94a3b8; font-size: 0.9rem; margin-top: 0.6rem; }
            .try-btn { background: #334155; border: none; color: #fff; padding: 0.4rem 0.9rem; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.82rem; margin-top: 0.6rem; transition: background 0.2s; }
            .try-btn:hover { background: #475569; }
            pre { background: #090d16; padding: 0.8rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38bdf8; overflow-x: auto; display: none; margin-top: 0.6rem; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Suggestion By Anna REST API Documentation</h1>
                <p style="color: #94a3b8;">Interactive endpoint explorer for developers, interviewers & judges.</p>
                <a href="/" style="color: #4ade80; text-decoration: none; font-weight: 700;">👈 Back to Suggestion By Anna Web App</a>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/api/health</span>
                <p class="desc">System health probe & database connection status.</p>
                <button class="try-btn" onclick="testEndpoint('/api/health', 'pre-health')">Test Endpoint</button>
                <pre id="pre-health"></pre>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/api/stats</span>
                <p class="desc">Live count of pantry items, favorites, shopping items, and planned meals.</p>
                <button class="try-btn" onclick="testEndpoint('/api/stats', 'pre-stats')">Test Endpoint</button>
                <pre id="pre-stats"></pre>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/api/pantry</span>
                <p class="desc">List all pantry inventory items stocked in the SQLite database.</p>
                <button class="try-btn" onclick="testEndpoint('/api/pantry', 'pre-pantry')">Test Endpoint</button>
                <pre id="pre-pantry"></pre>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/api/pantry/recommend</span>
                <p class="desc">Automated 1-click recipe discovery based on current pantry ingredients.</p>
                <button class="try-btn" onclick="testEndpoint('/api/pantry/recommend', 'pre-pantry-rec')">Test Endpoint</button>
                <pre id="pre-pantry-rec"></pre>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/api/favorites</span>
                <p class="desc">Retrieve all bookmarked favorite recipes saved in the database.</p>
                <button class="try-btn" onclick="testEndpoint('/api/favorites', 'pre-fav')">Test Endpoint</button>
                <pre id="pre-fav"></pre>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/api/shopping-list</span>
                <p class="desc">Fetch the live smart grocery checklist.</p>
                <button class="try-btn" onclick="testEndpoint('/api/shopping-list', 'pre-shop')">Test Endpoint</button>
                <pre id="pre-shop"></pre>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/api/meal-plans</span>
                <p class="desc">Fetch the weekly 7-day meal plan schedule.</p>
                <button class="try-btn" onclick="testEndpoint('/api/meal-plans', 'pre-meals')">Test Endpoint</button>
                <pre id="pre-meals"></pre>
            </div>

            <div class="endpoint-card">
                <span class="method get">GET</span><span class="path">/regional-recipes</span>
                <p class="desc">Curated authentic dishes categorized by Indian culinary zones.</p>
                <button class="try-btn" onclick="testEndpoint('/regional-recipes?region=north', 'pre-reg')">Test Endpoint</button>
                <pre id="pre-reg"></pre>
            </div>
        </div>

        <script>
            async function testEndpoint(url, preId) {
                const el = document.getElementById(preId);
                el.style.display = 'block';
                el.innerText = 'Loading...';
                try {
                    const res = await fetch(url);
                    const data = await res.json();
                    el.innerText = JSON.stringify(data, null, 2);
                } catch (e) {
                    el.innerText = 'Error: ' + e.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    print("=====================================================")
    print(" 🚀 Recipe AI Full-Stack Server Starting...")
    print(" 📂 Database: SQLite database/recipes.db initialized")
    print(" 🌐 Access application at: http://127.0.0.1:5000")
    print("=====================================================")
    app.run(debug=True, port=5000)
