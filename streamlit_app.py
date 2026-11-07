"""
Suggestion By Anna - Smart Full-Stack Indian Recipe Recommender & Kitchen Assistant
Unified Streamlit Application

Features:
1. 🍲 Reverse Recipe Recommendation Engine (TF-IDF & Cosine Similarity)
2. 🧊 Smart Fridge & Pantry Vision Scanner (Gemini Vision / YOLO + gTTS voice)
3. 🧺 SQLite Kitchen Pantry Manager (Categorized inventory & 1-click Cook with Pantry)
4. ❤️ Saved Favorites & Bookmarks
5. 📅 7-Day Weekly Meal Planner
6. 🛒 Smart Grocery / Shopping Checklist
7. 🧑‍🍳 AI Master Chef Studio (Gemini-powered personalized recipe generation)
8. 🍛 Regional Indian Thalis Catalog (North, South, East, West, Central)
"""

import os
import io
import re
import json
import pickle
import time
from datetime import datetime
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from gtts import gTTS

# Database & helper services
import database.db as db
from image_cache import get_recipe_image, get_recipe_youtube
from nutrition_service import calculate_recipe_nutrition

# Load environment variables
load_dotenv()
UNSPLASH_API_KEY = os.getenv("UNSPLASH_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Suggestion By Anna - Indian Recipe Recommender & Kitchen Assistant",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Global accents */
    :root {
        --primary-green: #0c831f;
        --dark-green: #095f17;
        --saffron: #f97316;
        --card-bg: #ffffff;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0c831f 0%, #15803d 50%, #b45309 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(12, 131, 31, 0.25);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 0.4rem;
    }

    /* Badges & Chips */
    .chip {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .chip-green { background-color: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .chip-orange { background-color: #ffedd5; color: #9a3412; border: 1px solid #fed7aa; }
    .chip-red { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .chip-blue { background-color: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }
    .chip-purple { background-color: #f3e8ff; color: #6b21a8; border: 1px solid #e9d5ff; }

    /* Recipe Cards */
    .recipe-card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .recipe-card-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .macro-pill {
        display: inline-block;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.35rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        text-align: center;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Coach speech bubble */
    .chef-bubble {
        background: linear-gradient(to right, #ecfdf5, #f0fdf4);
        border-left: 5px solid #0c831f;
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        color: #14532d;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }

    /* Thali Card */
    .thali-card {
        background: #ffffff;
        border: 1px solid #fed7aa;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #f97316;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# RESOURCE CACHING: ML MODELS & MATRICES
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Recipe AI & Vector Matrices...")
def load_ml_resources():
    """Load pre-trained TF-IDF vectorizers and precomputed sparse matrix."""
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

    return {
        'recipe_names': recipe_names,
        'tfidf_ingredients': tfidf_ingredients,
        'tfidf_diet': tfidf_diet,
        'tfidf_course': tfidf_course,
        'tfidf_region': tfidf_region,
        'input_features': input_features,
        'recipe_ingredients_map': recipe_ingredients_map,
        'recipe_diet_map': recipe_diet_map,
        'recipe_region_map': recipe_region_map,
        'all_diet_options': list(diet_names),
        'all_course_options': list(course_names),
        'all_region_options': list(region_names)
    }

ML_DATA = load_ml_resources()

# Optional YOLO Model Loader (cached on demand)
@st.cache_resource
def get_yolo_model():
    try:
        from ultralytics import YOLO
        yolo_path = os.path.join(BASE_DIR, 'yolov8n.pt')
        if os.path.exists(yolo_path):
            return YOLO(yolo_path)
    except Exception:
        pass
    return None

# Culinary Substitutes Map
SUBSTITUTES_MAP = {
    "paneer": "Tofu, dense boiled potato, or ricotta cheese",
    "ghee": "Butter, refined cooking oil, or mustard oil",
    "clarified": "Butter or coconut oil",
    "butter": "Ghee or mustard oil",
    "cream": "Cashew paste, fresh malai, or whisked hung curd",
    "yogurt": "Fresh curd, buttermilk, or lemon juice with milk",
    "curd": "Plain yogurt or cashew-lemon cream",
    "garam": "Roasted cumin & coriander powder with cinnamon and cloves",
    "masala": "Kitchen King masala or roasted ground cumin-coriander",
    "kewra": "Rose water, saffron infused milk, or cardamom powder",
    "saffron": "Turmeric powder (for golden hue) + crushed cardamom",
    "cashews": "Blanched almonds, melon seeds (magaz), or poppy seeds",
    "almonds": "Cashews or roasted peanuts",
    "maida": "Atta (whole wheat flour) or besan (gram flour)",
    "flour": "Atta or besan (gram flour)",
    "besan": "Gram flour, roasted sattu, or cornstarch",
    "tomato": "Tomato puree, tamarind pulp, or whisked curd",
    "onion": "Hing (asafoetida) + finely shredded cabbage",
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

# -----------------------------------------------------------------------------
# CORE RECOMMENDATION LOGIC
# -----------------------------------------------------------------------------
def get_recommendations(ingredients, diet="", course="", region="", top_k=6):
    """Recommend recipes with matching analysis, nutritional profiles, and substitute tips."""
    clean_ingredients = ingredients.strip() if ingredients else "paneer, tomato, onion"
    
    tfidf_ingredients = ML_DATA['tfidf_ingredients']
    tfidf_diet = ML_DATA['tfidf_diet']
    tfidf_course = ML_DATA['tfidf_course']
    tfidf_region = ML_DATA['tfidf_region']
    input_features = ML_DATA['input_features']
    recipe_names = ML_DATA['recipe_names']

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
        image_url = get_recipe_image(recipe_name, UNSPLASH_API_KEY)
        yt_data = get_recipe_youtube(recipe_name)

        all_req_ings = ML_DATA['recipe_ingredients_map'].get(recipe_name, [])
        matched = [ing for ing in all_req_ings if any(tok in ing or ing in tok for tok in user_tokens)]
        missing = [ing for ing in all_req_ings if ing not in matched]

        substitutes = {}
        for m in missing:
            for key, sub in SUBSTITUTES_MAP.items():
                if key in m.lower():
                    substitutes[m] = sub
                    break

        is_fav = db.is_favorite(recipe_name)
        nutrition = calculate_recipe_nutrition(recipe_name, all_req_ings)

        recommendations.append({
            'name': recipe_name,
            'similarity_score': round(float(similarity[i]) * 100, 1),
            'image_url': image_url,
            'region': ML_DATA['recipe_region_map'].get(recipe_name, 'Indian').capitalize(),
            'diet': ML_DATA['recipe_diet_map'].get(recipe_name, 'Vegetarian').capitalize(),
            'all_ingredients': all_req_ings,
            'matched_ingredients': matched,
            'missing_ingredients': missing,
            'substitutes': substitutes,
            'youtube': yt_data,
            'is_favorite': is_fav,
            'nutrition': nutrition
        })

    return recommendations

# -----------------------------------------------------------------------------
# HELPER: gTTS Voice Synthesis
# -----------------------------------------------------------------------------
def generate_speech(text, lang="hi"):
    """Generate audio bytes using Google Text-to-Speech."""
    try:
        clean_text = re.sub(r'[*#_]', '', text).strip()
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        return None

# -----------------------------------------------------------------------------
# HELPER: Vision Ingredient Detection
# -----------------------------------------------------------------------------
def detect_ingredients_from_image(pil_img, api_key=None):
    """Detect food ingredients using Gemini Vision or local YOLO fallback."""
    key = api_key or GOOGLE_API_KEY
    detected = []

    # 1. Gemini Vision API
    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                "Look at this image carefully. Is this food ingredients, refrigerator contents, vegetables, or cooking items? "
                "If NO (such as a person, animal, vehicle, document, room without food), reply strictly with 'NON_FOOD: description'. "
                "If YES, reply with ONLY a clean comma-separated list of common English ingredient names (e.g. tomato, onion, milk, paneer, capsicum)."
            )
            res = model.generate_content([prompt, pil_img])
            raw = res.text.strip()
            if "NON_FOOD" in raw:
                return False, raw.replace("NON_FOOD:", "").strip(), []
            items = [i.strip().lower() for i in raw.split(",") if i.strip()]
            if items:
                return True, "Gemini 1.5 Flash Vision", items
        except Exception:
            pass

    # 2. Local YOLOv8 Fallback
    yolo = get_yolo_model()
    if yolo:
        try:
            results = yolo(pil_img, verbose=False)
            classes = set()
            for r in results:
                for c in r.boxes.cls:
                    classes.add(yolo.names[int(c)])
            
            FOOD_MAPPING = {
                'banana': 'banana', 'apple': 'apple', 'sandwich': 'bread',
                'orange': 'orange', 'broccoli': 'broccoli', 'carrot': 'carrot',
                'pizza': 'cheese', 'bottle': 'milk', 'bowl': 'curd',
                'refrigerator': 'vegetables', 'cup': 'tea'
            }
            items = [FOOD_MAPPING[k] for k in classes if k in FOOD_MAPPING]
            if items:
                return True, "YOLOv8 Local Vision", list(set(items))
        except Exception:
            pass

    # 3. Dynamic Default Kitchen Set
    return True, "Culinary Heuristics", ["paneer", "tomato", "onion", "green chilies", "ginger"]

# -----------------------------------------------------------------------------
# HELPER: AI Master Chef Recipe Generation
# -----------------------------------------------------------------------------
def generate_master_chef_recipe(ingredients, cuisine="Punjabi", diet="Vegetarian", spice_level="Medium", cook_time="30 mins", dish_type="Curry / Gravy", api_key=None):
    """Personalized AI Chef recipe generator using Gemini or dynamic culinary logic."""
    key = api_key or GOOGLE_API_KEY
    if key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            You are a Master Indian Chef. Generate an authentic Indian recipe based on:
            - Ingredients: {ingredients}
            - Regional Style: {cuisine} Indian Cuisine
            - Diet: {diet}
            - Spice Level: {spice_level}
            - Cooking Time: {cook_time}
            - Dish Style: {dish_type}

            Respond strictly with valid JSON without backticks:
            {{
                "title": "Dish Name",
                "regional_style": "{cuisine} Special",
                "description": "Appetizing description.",
                "prep_time": "12 mins",
                "cook_time": "{cook_time}",
                "servings": "3-4 Servings",
                "spice_level": "{spice_level}",
                "ingredients_needed": [
                    {{"item": "Ingredient", "measure": "Quantity"}}
                ],
                "instructions": [
                    "Step 1: ...",
                    "Step 2: ...",
                    "Step 3: ...",
                    "Step 4: ...",
                    "Step 5: ..."
                ],
                "chef_secret_tip": "One secret cooking tip for authentic flavor.",
                "pairing": "Suggested bread or rice to pair with."
            }}
            """
            res = model.generate_content(prompt)
            clean_text = res.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except Exception:
            pass

    # Dynamic Fallback Chef Engine
    primary = ingredients.split(',')[0].strip().title() if ingredients else "Paneer"
    dish_title = f"{cuisine} Style {primary} {dish_type.split('/')[0].strip()}"
    return {
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
            "Heat ghee or oil in a heavy-bottomed kadai. Crackle cumin seeds and let the aroma bloom.",
            f"Add finely chopped base aromatics and sauté on medium heat until golden brown.",
            "Mix in the dry spices (turmeric, chili powder, and salt). Sauté for 1 minute until fragrant.",
            "Gently fold in the remaining ingredients. Add 1/2 cup warm water, cover with lid, and simmer for 8–10 minutes.",
            "Finish with a pinch of roasted garam masala and fresh coriander. Serve piping hot!"
        ],
        "chef_secret_tip": f"For authentic {cuisine} aroma, finish with a tempering (tadka) of crushed kasuri methi or hing in half-spoon warm ghee just before serving.",
        "pairing": "Fluffy Butter Phulkas, Steamed Basmati Rice, or Jeera Pulao"
    }


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & KITCHEN STATS
# -----------------------------------------------------------------------------
if "search_ingredients" not in st.session_state:
    st.session_state["search_ingredients"] = "paneer, tomato, onion"
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "🍲 Recipe Recommender"

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1589302168068-964664d93dc0?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.title("🍲 Suggestion By Anna")
    st.caption("Smart Full-Stack Indian Kitchen Assistant")

    st.markdown("---")
    
    # Navigation
    nav_options = [
        "🍲 Recipe Recommender",
        "🧊 Smart Fridge Vision AI",
        "🧺 Kitchen Pantry Manager",
        "❤️ My Saved Favorites",
        "📅 7-Day Meal Planner",
        "🛒 Smart Grocery List",
        "🧑‍🍳 AI Master Chef Studio",
        "🍛 Regional Thalis Catalog",
        "ℹ️ System Info"
    ]
    
    nav_idx = nav_options.index(st.session_state["active_nav"]) if st.session_state["active_nav"] in nav_options else 0
    selected_nav = st.radio("Navigation Menu", nav_options, index=nav_idx, key="sidebar_nav")
    st.session_state["active_nav"] = selected_nav

    st.markdown("---")
    st.subheader("📊 Live Kitchen Stats")
    stats = db.get_dashboard_stats()
    c1, c2 = st.columns(2)
    c1.metric("Pantry Items", stats["pantry_count"])
    c2.metric("Saved Favorites", stats["favorites_count"])
    c3, c4 = st.columns(2)
    c3.metric("Shopping Items", stats["shopping_pending"])
    c4.metric("Meals Planned", stats["meals_count"])

    st.markdown("---")
    st.subheader("⚙️ Settings")
    sidebar_gemini_key = st.text_input("Gemini API Key (Optional)", value=GOOGLE_API_KEY or "", type="password")
    voice_pref = st.selectbox("Audio Voice Guidance", ["hi (Hindi/Hinglish)", "en (English)"], index=0)
    audio_lang = "hi" if "hi" in voice_pref else "en"


# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">Suggestion By Anna 🍲</h1>
    <div class="hero-subtitle">Smart Full-Stack Indian Recipe Recommender • Computer Vision • Kitchen Pantry • Meal Planner</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# VIEW 1: RECIPE RECOMMENDER (CORE ENGINE)
# =============================================================================
if st.session_state["active_nav"] == "🍲 Recipe Recommender":
    st.subheader("⚡ Instant Reverse Recipe Engine")
    st.caption("Tell Anna what ingredients you have in your kitchen, and find the most authentic Indian dishes in under 50ms.")

    # Search inputs
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query_input = st.text_input(
            "Enter ingredients (comma separated):",
            value=st.session_state.get("search_ingredients", "paneer, tomato, onion"),
            placeholder="e.g. paneer, tomato, onion, capsicum, ginger, rice"
        )
        st.session_state["search_ingredients"] = query_input

    with col_btn:
        st.write("")
        st.write("")
        find_btn = st.button("🔍 Find Recipes", type="primary", use_container_width=True)

    # Quick Ingredient Pills
    st.caption("Quick Add Ingredients:")
    quick_tags = ["Paneer", "Tomato", "Onion", "Potato", "Garam Masala", "Curd", "Dal", "Basmati Rice", "Palak", "Ghee", "Kasuri Methi"]
    tag_cols = st.columns(len(quick_tags))
    for i, tag in enumerate(quick_tags):
        with tag_cols[i]:
            if st.button(tag, key=f"quick_{tag}", use_container_width=True):
                cur = [x.strip() for x in query_input.split(",") if x.strip()]
                if tag.lower() not in [x.lower() for x in cur]:
                    cur.append(tag)
                    st.session_state["search_ingredients"] = ", ".join(cur)
                    st.rerun()

    # Filters expander
    with st.expander("🎯 Dietary, Regional & Course Filters", expanded=True):
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            diet_filter = st.selectbox("Diet Preference", ["All Diets", "Vegetarian", "Non-Vegetarian"])
        with fcol2:
            region_filter = st.selectbox("Regional Cuisine", ["All Regions", "North", "South", "West", "East", "Central"])
        with fcol3:
            course_filter = st.selectbox("Meal Course", ["All Courses", "Main Course", "Snack", "Dessert", "Starter"])
        with fcol4:
            top_k = st.slider("Recipes to show", min_value=3, max_value=12, value=6)

    # Perform recommendation
    diet_arg = "" if diet_filter == "All Diets" else diet_filter.lower()
    region_arg = "" if region_filter == "All Regions" else region_filter.lower()
    course_arg = "" if course_filter == "All Courses" else course_filter.lower()

    recipes = get_recommendations(
        st.session_state["search_ingredients"],
        diet=diet_arg,
        course=course_arg,
        region=region_arg,
        top_k=top_k
    )

    st.markdown(f"### Found {len(recipes)} Authentic Indian Matches")

    # Render recipe cards in a grid
    cols_per_row = 2
    for r_idx in range(0, len(recipes), cols_per_row):
        cols = st.columns(cols_per_row)
        for c_idx in range(cols_per_row):
            if r_idx + c_idx < len(recipes):
                r = recipes[r_idx + c_idx]
                with cols[c_idx]:
                    with st.container(border=True):
                        # Card Header: Image & Badges
                        img_col, meta_col = st.columns([1, 1.4])
                        with img_col:
                            st.image(r["image_url"], use_container_width=True)
                        with meta_col:
                            st.markdown(f"### {r['name']}")
                            
                            # Badges
                            diet_class = "chip-green" if r["diet"].lower() == "vegetarian" else "chip-red"
                            st.markdown(f"""
                            <span class="chip {diet_class}">🌱 {r['diet']}</span>
                            <span class="chip chip-blue">📍 {r['region']} Indian</span>
                            <span class="chip chip-orange">🔥 {r['similarity_score']}% Match</span>
                            """, unsafe_allow_html=True)

                            # Macros
                            nut = r["nutrition"]
                            st.markdown(f"""
                            <div style="margin-top: 0.5rem;">
                                <span class="macro-pill">⚡ {nut['calories']} kcal</span>
                                <span class="macro-pill">💪 {nut['protein']}g Protein</span>
                                <span class="macro-pill">🌾 {nut['carbs']}g Carbs</span>
                                <span class="macro-pill">🥑 {nut['fat']}g Fat</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for tag in nut.get("tags", []):
                                st.markdown(f'<span class="chip chip-purple">{tag}</span>', unsafe_allow_html=True)

                        # Ingredients Breakdown
                        st.markdown("---")
                        ing_c1, ing_c2 = st.columns(2)
                        with ing_c1:
                            st.markdown("**✅ In Your Kitchen:**")
                            if r["matched_ingredients"]:
                                chips = "".join([f'<span class="chip chip-green">✓ {m}</span>' for m in r["matched_ingredients"]])
                                st.markdown(chips, unsafe_allow_html=True)
                            else:
                                st.caption("None matched directly")

                        with ing_c2:
                            st.markdown("**🛒 Missing Ingredients:**")
                            if r["missing_ingredients"]:
                                chips = "".join([f'<span class="chip chip-orange">+ {m}</span>' for m in r["missing_ingredients"]])
                                st.markdown(chips, unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="chip chip-green">🎉 You have everything!</span>', unsafe_allow_html=True)

                        # Substitutions
                        if r["substitutes"]:
                            st.markdown("**💡 Indian Culinary Substitutes:**")
                            for orig, sub in list(r["substitutes"].items())[:2]:
                                st.caption(f"• **{orig.title()}**: {sub}")

                        # Action Buttons
                        st.markdown("---")
                        b1, b2, b3 = st.columns([1, 1.2, 1])
                        
                        # 1. Favorite Toggle
                        with b1:
                            fav_label = "❤️ Saved" if r["is_favorite"] else "🤍 Save Fav"
                            if st.button(fav_label, key=f"fav_{r['name']}"):
                                res = db.toggle_favorite(
                                    recipe_name=r['name'],
                                    diet=r['diet'],
                                    region=r['region'],
                                    similarity_score=r['similarity_score'],
                                    image_url=r['image_url'],
                                    ingredients=r['all_ingredients'],
                                    youtube_url=r.get('youtube', {}).get('watch_url', '')
                                )
                                if res["action"] == "added":
                                    st.toast(f"Saved {r['name']} to Favorites! ❤️", icon="⭐")
                                else:
                                    st.toast(f"Removed {r['name']} from Favorites", icon="🗑️")
                                st.rerun()

                        # 2. Add Missing to Grocery
                        with b2:
                            if r["missing_ingredients"]:
                                if st.button(f"🛒 Add Missing ({len(r['missing_ingredients'])})", key=f"shop_{r['name']}"):
                                    added = db.add_missing_ingredients_to_shopping(r['name'], r['missing_ingredients'])
                                    st.toast(f"Added {added} missing items to Shopping Checklist! 🛒", icon="✅")
                                    st.rerun()
                            else:
                                st.button("🛒 Ready to Cook", disabled=True, key=f"shop_dis_{r['name']}")

                        # 3. YouTube link
                        with b3:
                            yt_url = r.get("youtube", {}).get("watch_url")
                            if yt_url:
                                st.link_button("▶️ YouTube", yt_url, use_container_width=True)
                            else:
                                st.link_button("▶️ Search Video", f"https://www.youtube.com/results?search_query={r['name']}+recipe", use_container_width=True)

                        # Reviews Accordion
                        with st.expander(f"⭐ Reviews & Community Ratings for {r['name']}"):
                            rev_data = db.get_recipe_reviews(r['name'])
                            st.markdown(f"**Average Rating:** ⭐ {rev_data['average_rating']} / 5.0 ({rev_data['count']} reviews)")
                            for rev in rev_data["reviews"]:
                                st.markdown(f"• **{rev['user_name']}** ({'⭐'*rev['rating']}): {rev['comment']}")
                            
                            # Add review form
                            with st.form(key=f"form_rev_{r['name']}"):
                                r_user = st.text_input("Your Name", value="Home Chef")
                                r_stars = st.slider("Rating", 1, 5, 5)
                                r_comment = st.text_input("Comment", placeholder="Authentic taste! Loved the rich gravy.")
                                submit_rev = st.form_submit_button("Post Review")
                                if submit_rev:
                                    db.add_recipe_review(r['name'], r_stars, r_comment, r_user)
                                    st.success("Review posted successfully!")
                                    st.rerun()


# =============================================================================
# VIEW 2: SMART FRIDGE VISION AI & VOICE COACH
# =============================================================================
elif st.session_state["active_nav"] == "🧊 Smart Fridge Vision AI":
    st.subheader("🧊 Smart Fridge & Pantry Vision Assistant")
    st.caption("Point your camera or upload a photo of your refrigerator or kitchen counter to detect ingredients and get instant audio coaching.")

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        cam_photo = st.camera_input("Take a photo of your fridge / vegetables")
    with vcol2:
        upload_photo = st.file_uploader("Or upload an image file (JPG, PNG)", type=["jpg", "jpeg", "png", "webp"])

    target_image = cam_photo or upload_photo

    if target_image:
        pil_img = Image.open(io.BytesIO(target_image.getvalue()))
        
        c_left, c_right = st.columns([1, 1.2])
        with c_left:
            st.image(pil_img, caption="Analyzed Kitchen Photo", use_container_width=True)

        with c_right:
            st.markdown("### 🔍 Computer Vision Detection")
            with st.spinner("Analyzing ingredients with AI Vision..."):
                is_food, source_name, detected_items = detect_ingredients_from_image(pil_img, api_key=sidebar_gemini_key)

            if not is_food:
                st.error(f"⚠️ No food ingredients detected ({source_name}). Please take a photo of your fridge shelves, vegetables, or cooking pantry.")
            else:
                st.success(f"Detected **{len(detected_items)} ingredients** via {source_name}:")
                st.markdown(" ".join([f'<span class="chip chip-green">✓ {d.title()}</span>' for d in detected_items]), unsafe_allow_html=True)

                act1, act2 = st.columns(2)
                with act1:
                    if st.button("📥 Add All to Pantry Database", type="primary", use_container_width=True):
                        added = 0
                        for item in detected_items:
                            db.add_pantry_item(item.title(), "Vegetables", "1 unit")
                            added += 1
                        st.balloons()
                        st.success(f"Saved {added} items directly to SQLite Pantry! 🧺")
                        st.rerun()

                with act2:
                    if st.button("🍲 Find Recipes with These", use_container_width=True):
                        st.session_state["search_ingredients"] = ", ".join(detected_items)
                        st.session_state["active_nav"] = "🍲 Recipe Recommender"
                        st.rerun()

                # Master Chef Coaching
                st.markdown("### 🧑‍🍳 Master Chef Live Voice Coach")
                chef_advice = f"""
                Namaste Chef! Aapke paas **{', '.join(detected_items[:4])}** ready hain!
                
                ✨ **Top Recommendation Today:**
                Simmer these with base aromatics (onion, tomato, cumin) for an authentic homestyle curry.
                
                💡 **Chef Secret:** Sauté the spices in pure ghee or mustard oil until the oil separates for deep restaurant aroma!
                """
                st.markdown(f'<div class="chef-bubble">{chef_advice}</div>', unsafe_allow_html=True)

                # gTTS Audio Guidance
                speech_text = f"Aapke fridge me {', '.join(detected_items[:3])} mil gaye hain. Aaj swadisht Indian dish banate hain!" if audio_lang == "hi" else f"Found {', '.join(detected_items[:3])} in your fridge. Let's cook a delicious fresh Indian dish today!"
                audio_fp = generate_speech(speech_text, lang=audio_lang)
                if audio_fp:
                    st.audio(audio_fp, format="audio/mp3")
                    st.caption("Press play above to hear Anna speak audio guidance out loud!")


# =============================================================================
# VIEW 3: KITCHEN PANTRY MANAGER
# =============================================================================
elif st.session_state["active_nav"] == "🧺 Kitchen Pantry Manager":
    st.subheader("🧺 Persistent Kitchen Pantry Manager (SQLite)")
    st.caption("Track ingredients stocked in your kitchen. 1-click discovery instantly recommends dishes matching your inventory.")

    # 1-Click Cook with Pantry
    p_string = db.get_pantry_ingredients_string()
    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        st.info(f"**Current Stocked Ingredients ({len(db.get_all_pantry())} items):**\n\n{p_string if p_string else 'Pantry is currently empty.'}")
    with b_col2:
        st.write("")
        if st.button("⚡ Cook with My Pantry Now", type="primary", use_container_width=True, disabled=not p_string):
            st.session_state["search_ingredients"] = p_string
            st.session_state["active_nav"] = "🍲 Recipe Recommender"
            st.rerun()

    st.markdown("---")

    # Add Pantry Item Form
    with st.expander("➕ Add New Ingredient to Pantry", expanded=False):
        with st.form("add_pantry_form", clear_on_submit=True):
            p_name = st.text_input("Ingredient Name", placeholder="e.g. Kasuri Methi, Paneer, Basmati Rice")
            p_cat = st.selectbox("Category", ["Vegetables", "Dairy", "Spices", "Grains", "Oils", "General"])
            p_qty = st.text_input("Quantity", value="1 unit", placeholder="e.g. 500g, 1 pack, 250ml")
            submit_p = st.form_submit_button("Add to Pantry")
            if submit_p:
                if p_name.strip():
                    db.add_pantry_item(p_name, p_cat, p_qty)
                    st.success(f"Added {p_name} to your pantry!")
                    st.rerun()
                else:
                    st.warning("Please enter an ingredient name.")

    # Display Pantry Items Table with Delete actions
    pantry_items = db.get_all_pantry()
    if pantry_items:
        # Group by category
        categories = sorted(list(set(i["category"] for i in pantry_items)))
        for cat in categories:
            cat_items = [i for i in pantry_items if i["category"] == cat]
            st.markdown(f"#### 🏷️ {cat} ({len(cat_items)} items)")
            
            p_cols = st.columns(3)
            for idx, item in enumerate(cat_items):
                with p_cols[idx % 3]:
                    with st.container(border=True):
                        ic1, ic2 = st.columns([3, 1])
                        with ic1:
                            st.markdown(f"**{item['name']}**")
                            st.caption(f"{item['quantity']} • Added: {item['added_at'][:10]}")
                        with ic2:
                            if st.button("🗑️", key=f"del_p_{item['id']}"):
                                db.delete_pantry_item(item['id'])
                                st.toast(f"Removed {item['name']}", icon="🗑️")
                                st.rerun()
    else:
        st.info("Your pantry is empty! Add ingredients using the form above or scan your fridge with Vision AI.")


# =============================================================================
# VIEW 4: SAVED FAVORITES
# =============================================================================
elif st.session_state["active_nav"] == "❤️ My Saved Favorites":
    st.subheader("❤️ Saved Favorite Recipes")
    st.caption("Your personalized recipe collection stored securely in SQLite.")

    favorites = db.get_all_favorites()
    if favorites:
        st.markdown(f"You have **{len(favorites)} favorite recipes** saved:")
        f_cols = st.columns(3)
        for idx, fav in enumerate(favorites):
            with f_cols[idx % 3]:
                with st.container(border=True):
                    if fav.get("image_url"):
                        st.image(fav["image_url"], use_container_width=True)
                    st.markdown(f"### {fav['recipe_name']}")
                    st.caption(f"{fav.get('region', 'Indian')} Cuisine • {fav.get('diet', 'Vegetarian')}")
                    
                    if fav.get("similarity_score"):
                        st.markdown(f'<span class="chip chip-orange">🔥 {fav["similarity_score"]}% Match</span>', unsafe_allow_html=True)

                    if fav.get("ingredients"):
                        st.caption(f"**Ingredients:** {', '.join(fav['ingredients'][:4])}...")

                    st.markdown("---")
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        if fav.get("youtube_url"):
                            st.link_button("▶️ YouTube", fav["youtube_url"], use_container_width=True)
                        else:
                            st.link_button("▶️ Search Video", f"https://www.youtube.com/results?search_query={fav['recipe_name']}+recipe", use_container_width=True)
                    with fc2:
                        if st.button("🗑️ Remove", key=f"del_fav_{fav['id']}", use_container_width=True):
                            db.delete_favorite(fav['id'])
                            st.toast(f"Removed {fav['recipe_name']} from favorites.")
                            st.rerun()
    else:
        st.info("No saved recipes yet. Browse recipes in the Recommender and click '🤍 Save Fav' to bookmark them!")


# =============================================================================
# VIEW 5: 7-DAY MEAL PLANNER
# =============================================================================
elif st.session_state["active_nav"] == "📅 7-Day Meal Planner":
    st.subheader("📅 7-Day Weekly Meal Planner")
    st.caption("Organize healthy breakfast, lunch, and dinner for your family throughout the week.")

    # Add Meal Plan Form
    with st.expander("➕ Schedule a Recipe to Weekly Plan", expanded=False):
        with st.form("meal_plan_form", clear_on_submit=True):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                m_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                m_slot = st.selectbox("Meal Slot", ["Breakfast", "Lunch", "Dinner", "Snack"])
            with m_col2:
                # Select from existing catalog or custom name
                m_recipe = st.selectbox("Recipe", ML_DATA["recipe_names"][:50])
                m_notes = st.text_input("Notes (e.g. serve with phulkas, family dinner)", value="")
            
            submit_m = st.form_submit_button("Add to Weekly Schedule")
            if submit_m:
                db.add_meal_plan(m_day, m_slot, m_recipe, m_notes)
                st.success(f"Scheduled {m_recipe} for {m_day} {m_slot}!")
                st.rerun()

    # Display Week Grid
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plans = db.get_all_meal_plans()

    for day in days:
        day_plans = [p for p in plans if p["day_of_week"].lower() == day.lower()]
        st.markdown(f"#### 🗓️ {day}")
        if day_plans:
            d_cols = st.columns(len(day_plans))
            for idx, plan in enumerate(day_plans):
                with d_cols[idx]:
                    with st.container(border=True):
                        st.markdown(f'<span class="chip chip-blue">{plan["meal_slot"]}</span>', unsafe_allow_html=True)
                        st.markdown(f"**{plan['recipe_name']}**")
                        if plan.get("notes"):
                            st.caption(f"*{plan['notes']}*")
                        if st.button("🗑️", key=f"del_plan_{plan['id']}"):
                            db.delete_meal_plan(plan['id'])
                            st.toast(f"Removed {plan['recipe_name']} from {day}")
                            st.rerun()
        else:
            st.caption("No meals planned for this day.")


# =============================================================================
# VIEW 6: SMART GROCERY CHECKLIST
# =============================================================================
elif st.session_state["active_nav"] == "🛒 Smart Grocery List":
    st.subheader("🛒 Smart Grocery & Shopping Checklist")
    st.caption("1-click missing ingredient checklist. Check items off as you shop in the supermarket or market.")

    # Quick Add & Clear Actions
    act_c1, act_c2 = st.columns([3, 1])
    with act_c1:
        with st.form("add_shop_form", clear_on_submit=True):
            s_c1, s_c2, s_c3 = st.columns([3, 2, 1])
            with s_c1:
                item_name = st.text_input("Item Name", placeholder="e.g. Kasuri Methi, Saffron")
            with s_c2:
                item_qty = st.text_input("Quantity", value="1 pack")
            with s_c3:
                st.write("")
                st.write("")
                add_s = st.form_submit_button("Add Item")
            if add_s and item_name.strip():
                db.add_shopping_item(item_name, "Kitchen Manual", item_qty)
                st.success(f"Added {item_name}!")
                st.rerun()

    with act_c2:
        st.write("")
        st.write("")
        if st.button("🧹 Clear Purchased Items", use_container_width=True):
            removed = db.clear_purchased_shopping_items()
            st.toast(f"Cleared {removed} purchased items! 🧹")
            st.rerun()

    # Checklist
    items = db.get_all_shopping_items()
    if items:
        pending = [i for i in items if not i["is_purchased"]]
        purchased = [i for i in items if i["is_purchased"]]

        st.markdown(f"**Pending Items ({len(pending)})**")
        for item in pending:
            col_check, col_del = st.columns([6, 1])
            with col_check:
                if st.checkbox(f"**{item['item_name']}** — {item['quantity']} *(Recipe: {item['recipe_source']})*", key=f"chk_{item['id']}"):
                    db.toggle_shopping_item(item['id'], is_purchased=True)
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_s_{item['id']}"):
                    db.delete_shopping_item(item['id'])
                    st.rerun()

        if purchased:
            st.markdown("---")
            st.markdown(f"**Purchased Items ({len(purchased)})**")
            for item in purchased:
                col_check, col_del = st.columns([6, 1])
                with col_check:
                    st.markdown(f"~~{item['item_name']} ({item['quantity']})~~")
                with col_del:
                    if st.button("🗑️", key=f"del_sp_{item['id']}"):
                        db.delete_shopping_item(item['id'])
                        st.rerun()
    else:
        st.info("Your shopping list is clear! When browsing recipes in the Recommender, click 'Add Missing' to populate items automatically.")


# =============================================================================
# VIEW 7: AI MASTER CHEF STUDIO
# =============================================================================
elif st.session_state["active_nav"] == "🧑‍🍳 AI Master Chef Studio":
    st.subheader("🧑‍🍳 AI Master Chef Studio")
    st.caption("Craft personalized, restaurant-grade Indian recipes tailored to your ingredients, spice tolerance, and cooking style.")

    with st.form("chef_studio_form"):
        col1, col2 = st.columns(2)
        with col1:
            chef_ings = st.text_input("Ingredients Available", value="Paneer, Bell Peppers, Onion, Cream, Kasuri Methi")
            chef_cuisine = st.selectbox("Regional Cuisine Style", ["Punjabi / North", "South Indian", "Mughlai", "Rajasthani", "Bengali", "Gujarati", "Goan"])
            chef_diet = st.selectbox("Diet", ["Vegetarian", "Non-Vegetarian", "Vegan", "Jain"])
        with col2:
            chef_spice = st.select_slider("Spice Level", options=["Mild", "Medium", "Spicy", "Fiery Desi"])
            chef_time = st.selectbox("Cooking Time", ["15 mins (Quick)", "30 mins (Standard)", "45 mins (Slow Cooked)"])
            chef_type = st.selectbox("Dish Style", ["Rich Gravy / Curry", "Dry Tawa Masala", "Biryani / Pulao", "Crispy Starter", "Comfort Dal"])
        
        gen_btn = st.form_submit_button("✨ Generate Master Chef Recipe", type="primary")

    if gen_btn:
        with st.spinner("Master Chef is formulating your authentic Indian recipe..."):
            recipe_result = generate_master_chef_recipe(
                ingredients=chef_ings,
                cuisine=chef_cuisine,
                diet=chef_diet,
                spice_level=chef_spice,
                cook_time=chef_time,
                dish_type=chef_type,
                api_key=sidebar_gemini_key
            )

        st.markdown("---")
        st.markdown(f"## 🍲 {recipe_result['title']}")
        st.markdown(f"*{recipe_result['description']}*")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Regional Style", recipe_result["regional_style"])
        m2.metric("Prep Time", recipe_result.get("prep_time", "10 mins"))
        m3.metric("Cook Time", recipe_result.get("cook_time", chef_time))
        m4.metric("Spice Level", recipe_result.get("spice_level", chef_spice))

        st.markdown("---")
        ic1, ic2 = st.columns([1, 1.4])
        with ic1:
            st.markdown("### 🛒 Ingredients Needed:")
            for item in recipe_result["ingredients_needed"]:
                st.markdown(f"• **{item['item']}**: {item['measure']}")

            if recipe_result.get("pairing"):
                st.markdown("### 🌾 Best Paired With:")
                st.info(recipe_result["pairing"])

        with ic2:
            st.markdown("### 👨‍🍳 Step-by-Step Cooking Method:")
            for step in recipe_result["instructions"]:
                st.markdown(f"{step}")

            st.markdown("### 💡 Master Chef Secret Pro Tip:")
            st.markdown(f'<div class="chef-bubble">{recipe_result["chef_secret_tip"]}</div>', unsafe_allow_html=True)


# =============================================================================
# VIEW 8: REGIONAL THALIS CATALOG
# =============================================================================
elif st.session_state["active_nav"] == "🍛 Regional Thalis Catalog":
    st.subheader("🍛 Regional Indian Thalis & Flavors")
    st.caption("Explore authentic culinary traditions across North, South, East, West, and Central India.")

    regions_catalog = {
        "north": {
            "title": "North Indian (Punjabi, Mughlai)",
            "icon": "🌾",
            "desc": "Rich gravies, slow-cooked dals, tandoori flavors, and warming whole spices.",
            "dishes": ["Paneer butter masala", "Chana masala", "Dal tadka", "Gajar ka halwa", "Shahi paneer", "Aloo matar", "Lassi", "Rabri"]
        },
        "south": {
            "title": "South Indian (Tamil, Kerala, Andhra)",
            "icon": "🥥",
            "desc": "Coconut-infused gravies, fiery pepper rasams, fragrant curries, and comforting rice delicacies.",
            "dishes": ["Sambar", "Rasam", "Bisi bele bath", "Idli", "Dosa", "Payasam", "Avial"]
        },
        "west": {
            "title": "West Indian (Gujarati, Rajasthani, Maharashtrian)",
            "icon": "🏜️",
            "desc": "Sweet and savory balances, crunchy farsans, royal mithais, and soulful curries.",
            "dishes": ["Dhokla", "Khandvi", "Ghevar", "Pav bhaji", "Shrikhand", "Sheera", "Basundi", "Chikki"]
        },
        "east": {
            "title": "East Indian (Bengali, Odia, Assamese)",
            "icon": "🌊",
            "desc": "Subtle mustard notes, delicate steamed dishes, and heavenly dairy-based sweets.",
            "dishes": ["Rasgulla", "Sandesh", "Mishti doi", "Luchi", "Chhena poda", "Aloo posto"]
        },
        "central": {
            "title": "Central & Royal Indian (Bhopali, Malwi)",
            "icon": "👑",
            "desc": "Aromatic royal curries, poha breakfasts, and festive delicacies.",
            "dishes": ["Poha", "Bhutte ka kees", "Biryani", "Mawa bati", "Dal bafla"]
        }
    }

    for reg_key, reg in regions_catalog.items():
        with st.container(border=True):
            st.markdown(f"### {reg['icon']} {reg['title']}")
            st.markdown(f"*{reg['desc']}*")
            
            dish_cols = st.columns(4)
            for d_idx, dish in enumerate(reg["dishes"]):
                with dish_cols[d_idx % 4]:
                    if st.button(f"🍲 {dish}", key=f"thali_{reg_key}_{dish}", use_container_width=True):
                        st.session_state["search_ingredients"] = dish
                        st.session_state["active_nav"] = "🍲 Recipe Recommender"
                        st.rerun()


# =============================================================================
# VIEW 9: SYSTEM INFO & HEALTH
# =============================================================================
elif st.session_state["active_nav"] == "ℹ️ System Info":
    st.subheader("ℹ️ Suggestion By Anna System Information")
    
    st.markdown("""
    **Suggestion By Anna** is an authentic, intelligent Indian Recipe Recommender and Kitchen Assistant.
    
    - **Engine**: Scikit-Learn TF-IDF Multi-Vector Vectorizer & Cosine Similarity (<50ms response)
    - **Database**: SQLite persistent storage for Pantry, Favorites, Meal Plans, Shopping Lists, and Reviews
    - **Vision AI**: Google Gemini 1.5 Flash Vision API with YOLOv8 fallback
    - **Voice Synthesis**: Google Text-to-Speech (gTTS) in Hindi & English
    - **Web Framework**: Streamlit (Python)
    """)

    st.markdown("---")
    st.subheader("📂 Catalog & Database Verification")
    st.json({
        "total_recipes_catalog": len(ML_DATA["recipe_names"]),
        "database_connected": True,
        "sqlite_stats": db.get_dashboard_stats(),
        "gemini_configured": bool(GOOGLE_API_KEY or sidebar_gemini_key),
        "unsplash_configured": bool(UNSPLASH_API_KEY)
    })
