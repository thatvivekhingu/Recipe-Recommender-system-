"""
Nutrition & Dietary Intelligence Service for SwadAI.
Calculates approximate calorie count, macronutrients (Protein, Carbs, Fat), 
and dietary wellness tags for authentic Indian dishes based on ingredient composition.
"""

# Base macronutrient profiles per ingredient (approx per 100g portion contribution)
INGREDIENT_MACROS = {
    # Proteins & Dairy
    "paneer": {"cal": 265, "protein": 18, "carbs": 3, "fat": 20},
    "milk": {"cal": 60, "protein": 3.2, "carbs": 5, "fat": 3.5},
    "curd": {"cal": 60, "protein": 3.5, "carbs": 4.5, "fat": 3},
    "yogurt": {"cal": 60, "protein": 3.5, "carbs": 4.5, "fat": 3},
    "cheese": {"cal": 300, "protein": 20, "carbs": 2, "fat": 24},
    "chickpeas": {"cal": 164, "protein": 9, "carbs": 27, "fat": 2.5},
    "chana": {"cal": 164, "protein": 9, "carbs": 27, "fat": 2.5},
    "dal": {"cal": 116, "protein": 9, "carbs": 20, "fat": 0.5},
    "lentils": {"cal": 116, "protein": 9, "carbs": 20, "fat": 0.5},
    "rajma": {"cal": 127, "protein": 8.7, "carbs": 22.8, "fat": 0.5},
    
    # Grains & Flours
    "rice": {"cal": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
    "basmati": {"cal": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
    "flour": {"cal": 364, "protein": 10, "carbs": 76, "fat": 1},
    "atta": {"cal": 340, "protein": 12, "carbs": 72, "fat": 1.5},
    "besan": {"cal": 387, "protein": 22, "carbs": 58, "fat": 6.7},
    "maida": {"cal": 364, "protein": 10, "carbs": 76, "fat": 1},
    
    # Fats & Oils
    "ghee": {"cal": 900, "protein": 0, "carbs": 0, "fat": 100},
    "butter": {"cal": 717, "protein": 0.8, "carbs": 0.1, "fat": 81},
    "oil": {"cal": 884, "protein": 0, "carbs": 0, "fat": 100},
    "mustard": {"cal": 150, "protein": 4, "carbs": 8, "fat": 12},
    "cream": {"cal": 340, "protein": 2.8, "carbs": 3.7, "fat": 36},
    "coconut": {"cal": 354, "protein": 3.3, "carbs": 15, "fat": 33},
    "cashews": {"cal": 553, "protein": 18, "carbs": 30, "fat": 44},
    "almonds": {"cal": 579, "protein": 21, "carbs": 22, "fat": 49},
    
    # Vegetables & Aromatics
    "potato": {"cal": 77, "protein": 2, "carbs": 17, "fat": 0.1},
    "potatoes": {"cal": 77, "protein": 2, "carbs": 17, "fat": 0.1},
    "tomato": {"cal": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
    "onion": {"cal": 40, "protein": 1.1, "carbs": 9.3, "fat": 0.1},
    "spinach": {"cal": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4},
    "palak": {"cal": 23, "protein": 2.9, "carbs": 3.6, "fat": 0.4},
    "carrot": {"cal": 41, "protein": 0.9, "carbs": 9.6, "fat": 0.2},
    "peas": {"cal": 81, "protein": 5.4, "carbs": 14.5, "fat": 0.4},
    "matar": {"cal": 81, "protein": 5.4, "carbs": 14.5, "fat": 0.4},
    "cauliflower": {"cal": 25, "protein": 1.9, "carbs": 5, "fat": 0.3},
    "gobi": {"cal": 25, "protein": 1.9, "carbs": 5, "fat": 0.3},
    
    # Sweets & Sugars
    "sugar": {"cal": 387, "protein": 0, "carbs": 100, "fat": 0},
    "jaggery": {"cal": 383, "protein": 0.4, "carbs": 98, "fat": 0.1}
}


def calculate_recipe_nutrition(recipe_name, ingredients_list):
    """Estimate calories, macros, and wellness tags for a given dish."""
    name_lower = recipe_name.lower()
    total_cal = 220
    protein = 7
    carbs = 24
    fat = 10

    # Match ingredients to compute cumulative values
    matched_count = 0
    for ing in ingredients_list:
        ing_lower = ing.lower()
        for key, vals in INGREDIENT_MACROS.items():
            if key in ing_lower:
                total_cal += vals["cal"] * 0.35
                protein += vals["protein"] * 0.35
                carbs += vals["carbs"] * 0.35
                fat += vals["fat"] * 0.35
                matched_count += 1
                break

    # Adjust based on dish types
    if any(s in name_lower for s in ["halwa", "jamun", "jalebi", "laddu", "sweet", "barfi", "kheer", "rabri"]):
        total_cal = max(total_cal, 340)
        carbs = max(carbs, 45)
        fat = max(fat, 14)
    elif any(p in name_lower for p in ["paneer", "kofta", "shahi"]):
        total_cal = max(total_cal, 310)
        protein = max(protein, 16)
        fat = max(fat, 18)
    elif any(d in name_lower for d in ["dal", "chana", "rajma"]):
        total_cal = max(total_cal, 210)
        protein = max(protein, 14)
        fat = min(fat, 8)

    # Round macros
    calories = int(round(total_cal))
    protein_g = round(protein, 1)
    carbs_g = round(carbs, 1)
    fat_g = round(fat, 1)

    # Determine Health / Wellness Tags
    tags = []
    if protein_g >= 14:
        tags.append({"label": "High Protein", "color": "#16a34a", "icon": "fa-dumbbell"})
    if calories <= 250:
        tags.append({"label": "Low Calorie", "color": "#0284c7", "icon": "fa-heart-pulse"})
    if carbs_g <= 20:
        tags.append({"label": "Low Carb", "color": "#8b5cf6", "icon": "fa-leaf"})
    if any(s in name_lower for s in ["dal", "tadka", "palak", "chana", "salad"]):
        tags.append({"label": "Fiber Rich", "color": "#059669", "icon": "fa-wheat-awn"})
    if not tags:
        tags.append({"label": "Comfort Food", "color": "#d97706", "icon": "fa-kitchen-set"})

    return {
        "calories": calories,
        "protein": f"{protein_g}g",
        "carbs": f"{carbs_g}g",
        "fat": f"{fat_g}g",
        "tags": tags
    }
