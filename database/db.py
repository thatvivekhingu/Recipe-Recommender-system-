import sqlite3
import os
import json
from datetime import datetime

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "recipes.db")


def get_connection():
    """Create a connection to the SQLite database with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all database tables and seed starter data if empty."""
    os.makedirs(DB_DIR, exist_ok=True)
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Pantry items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pantry_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE,
                category TEXT DEFAULT 'General',
                quantity TEXT DEFAULT '1 unit',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Favorite recipes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS favorite_recipes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                diet TEXT,
                region TEXT,
                similarity_score REAL,
                image_url TEXT,
                ingredients_json TEXT,
                youtube_url TEXT,
                notes TEXT,
                saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Shopping list table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shopping_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT NOT NULL COLLATE NOCASE,
                recipe_source TEXT DEFAULT 'Kitchen',
                quantity TEXT DEFAULT '1 pack',
                is_purchased INTEGER DEFAULT 0,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 4. Weekly meal planner table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meal_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                day_of_week TEXT NOT NULL,
                meal_slot TEXT NOT NULL,
                recipe_name TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 5. Recipe reviews table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recipe_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_name TEXT NOT NULL COLLATE NOCASE,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                comment TEXT,
                user_name TEXT DEFAULT 'Home Chef',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Seed starter data if pantry is empty
        cursor.execute("SELECT COUNT(*) FROM pantry_items")
        if cursor.fetchone()[0] == 0:
            starter_pantry = [
                ("Paneer", "Dairy", "250g"),
                ("Tomato", "Vegetables", "500g"),
                ("Onion", "Vegetables", "1kg"),
                ("Ghee", "Dairy", "200ml"),
                ("Garam Masala", "Spices", "100g"),
                ("Cumin Seeds", "Spices", "100g"),
                ("Turmeric Powder", "Spices", "100g"),
                ("Basmati Rice", "Grains", "1kg"),
                ("Curd / Yogurt", "Dairy", "400g"),
                ("Ginger", "Vegetables", "100g")
            ]
            cursor.executemany(
                "INSERT INTO pantry_items (name, category, quantity) VALUES (?, ?, ?)",
                starter_pantry
            )

        # Seed starter meal plan if empty
        cursor.execute("SELECT COUNT(*) FROM meal_plans")
        if cursor.fetchone()[0] == 0:
            starter_meals = [
                ("Monday", "Lunch", "Paneer butter masala", "Family favorite!"),
                ("Monday", "Dinner", "Dal tadka", "Light & comforting"),
                ("Tuesday", "Lunch", "Chana masala", "Serve with steamed rice"),
                ("Wednesday", "Dinner", "Aloo matar", "Quick 20 mins preparation"),
                ("Friday", "Dinner", "Shahi paneer", "Weekend special feast")
            ]
            cursor.executemany(
                "INSERT INTO meal_plans (day_of_week, meal_slot, recipe_name, notes) VALUES (?, ?, ?, ?)",
                starter_meals
            )

        # Seed starter shopping items if empty
        cursor.execute("SELECT COUNT(*) FROM shopping_list")
        if cursor.fetchone()[0] == 0:
            starter_shopping = [
                ("Kasuri Methi", "Paneer butter masala", "1 pack", 0),
                ("Fresh Coriander", "Dal tadka", "1 bunch", 0),
                ("Cashews / Kaju", "Shahi paneer", "100g", 0)
            ]
            cursor.executemany(
                "INSERT INTO shopping_list (item_name, recipe_source, quantity, is_purchased) VALUES (?, ?, ?, ?)",
                starter_shopping
            )

        conn.commit()


# ==========================================
# PANTRY CRUD
# ==========================================
def get_all_pantry():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pantry_items ORDER BY id DESC")
        return [dict(row) for row in cursor.fetchall()]


def add_pantry_item(name, category="General", quantity="1 unit"):
    clean_name = name.strip()
    if not clean_name:
        return None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO pantry_items (name, category, quantity) VALUES (?, ?, ?)",
            (clean_name, category.strip() or "General", quantity.strip() or "1 unit")
        )
        conn.commit()
        return cursor.lastrowid


def delete_pantry_item(item_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pantry_items WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_pantry_ingredients_string():
    """Return comma-separated string of all pantry ingredients for the recommender."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM pantry_items")
        items = [row["name"] for row in cursor.fetchall()]
        return ", ".join(items)


# ==========================================
# FAVORITES CRUD
# ==========================================
def get_all_favorites():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM favorite_recipes ORDER BY saved_at DESC")
        rows = cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["ingredients"] = json.loads(d.get("ingredients_json") or "[]")
            except Exception:
                d["ingredients"] = []
            results.append(d)
        return results


def is_favorite(recipe_name):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM favorite_recipes WHERE recipe_name = ?", (recipe_name.strip(),))
        return cursor.fetchone() is not None


def toggle_favorite(recipe_name, diet="", region="", similarity_score=0.0, image_url="", ingredients=None, youtube_url="", notes=""):
    name = recipe_name.strip()
    if not name:
        return {"success": False, "error": "Empty recipe name"}

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM favorite_recipes WHERE recipe_name = ?", (name,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute("DELETE FROM favorite_recipes WHERE id = ?", (existing["id"],))
            conn.commit()
            return {"success": True, "action": "removed", "is_favorite": False}
        else:
            ing_json = json.dumps(ingredients or [])
            cursor.execute("""
                INSERT INTO favorite_recipes 
                (recipe_name, diet, region, similarity_score, image_url, ingredients_json, youtube_url, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, diet, region, similarity_score, image_url, ing_json, youtube_url, notes))
            conn.commit()
            return {"success": True, "action": "added", "is_favorite": True, "id": cursor.lastrowid}


def delete_favorite(fav_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM favorite_recipes WHERE id = ?", (fav_id,))
        conn.commit()
        return cursor.rowcount > 0


# ==========================================
# SHOPPING LIST CRUD
# ==========================================
def get_all_shopping_items():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM shopping_list ORDER BY is_purchased ASC, id DESC")
        return [dict(row) for row in cursor.fetchall()]


def add_shopping_item(item_name, recipe_source="Kitchen", quantity="1 pack"):
    clean_name = item_name.strip()
    if not clean_name:
        return None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO shopping_list (item_name, recipe_source, quantity) VALUES (?, ?, ?)",
            (clean_name, recipe_source.strip() or "Kitchen", quantity.strip() or "1 pack")
        )
        conn.commit()
        return cursor.lastrowid


def add_missing_ingredients_to_shopping(recipe_name, missing_list):
    """Add multiple missing ingredients from a recipe to the shopping list."""
    if not missing_list:
        return 0
    added_count = 0
    with get_connection() as conn:
        cursor = conn.cursor()
        for item in missing_list:
            item_clean = item.strip().title()
            if item_clean:
                # Avoid duplicate unpurchased items for this recipe
                cursor.execute(
                    "SELECT id FROM shopping_list WHERE item_name = ? AND is_purchased = 0",
                    (item_clean,)
                )
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO shopping_list (item_name, recipe_source, quantity) VALUES (?, ?, ?)",
                        (item_clean, recipe_name, "1 unit")
                    )
                    added_count += 1
        conn.commit()
    return added_count


def toggle_shopping_item(item_id, is_purchased=None):
    with get_connection() as conn:
        cursor = conn.cursor()
        if is_purchased is None:
            cursor.execute("UPDATE shopping_list SET is_purchased = 1 - is_purchased WHERE id = ?", (item_id,))
        else:
            val = 1 if is_purchased else 0
            cursor.execute("UPDATE shopping_list SET is_purchased = ? WHERE id = ?", (val, item_id))
        conn.commit()
        return cursor.rowcount > 0


def delete_shopping_item(item_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM shopping_list WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_purchased_shopping_items():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM shopping_list WHERE is_purchased = 1")
        conn.commit()
        return cursor.rowcount


# ==========================================
# MEAL PLANNER CRUD
# ==========================================
def get_all_meal_plans():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM meal_plans ORDER BY id ASC")
        return [dict(row) for row in cursor.fetchall()]


def add_meal_plan(day_of_week, meal_slot, recipe_name, notes=""):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO meal_plans (day_of_week, meal_slot, recipe_name, notes) VALUES (?, ?, ?, ?)",
            (day_of_week.strip(), meal_slot.strip(), recipe_name.strip(), notes.strip())
        )
        conn.commit()
        return cursor.lastrowid


def delete_meal_plan(plan_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM meal_plans WHERE id = ?", (plan_id,))
        conn.commit()
        return cursor.rowcount > 0


# ==========================================
# REVIEWS & RATINGS CRUD
# ==========================================
def get_recipe_reviews(recipe_name):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM recipe_reviews WHERE recipe_name = ? ORDER BY created_at DESC",
            (recipe_name.strip(),)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        avg_rating = round(sum(r["rating"] for r in rows) / len(rows), 1) if rows else 5.0
        return {"reviews": rows, "count": len(rows), "average_rating": avg_rating}


def add_recipe_review(recipe_name, rating, comment="", user_name="Home Chef"):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO recipe_reviews (recipe_name, rating, comment, user_name) VALUES (?, ?, ?, ?)",
            (recipe_name.strip(), max(1, min(5, int(rating))), comment.strip(), user_name.strip() or "Home Chef")
        )
        conn.commit()
        return cursor.lastrowid


# ==========================================
# STATS COUNTER FOR DASHBOARD
# ==========================================
def get_dashboard_stats():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pantry_items")
        pantry_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM favorite_recipes")
        favorites_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM shopping_list WHERE is_purchased = 0")
        shopping_pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM meal_plans")
        meals_count = cursor.fetchone()[0]

        return {
            "pantry_count": pantry_count,
            "favorites_count": favorites_count,
            "shopping_pending": shopping_pending,
            "meals_count": meals_count,
            "total_recipes_catalog": 195
        }


# Auto-initialize database on import
init_db()
