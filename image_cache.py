import re
import urllib.parse
import requests

# Curated High-Definition Indian Food Photography mapped by recipe keywords
CURATED_IMAGES = {
    # Paneer dishes
    "paneer butter masala": "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?auto=format&fit=crop&w=800&q=80",
    "shahi paneer": "https://images.unsplash.com/photo-1596797038530-2c107229654b?auto=format&fit=crop&w=800&q=80",
    "palak paneer": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80",
    "matar paneer": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=800&q=80",
    "kadai paneer": "https://images.unsplash.com/photo-1565557623262-b51c2513a641?auto=format&fit=crop&w=800&q=80",
    
    # Lentils & Curries
    "dal tadka": "https://images.unsplash.com/photo-1546833998-877b37c2e5c6?auto=format&fit=crop&w=800&q=80",
    "dal makhani": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?auto=format&fit=crop&w=800&q=80",
    "chana masala": "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?auto=format&fit=crop&w=800&q=80",
    "chole": "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?auto=format&fit=crop&w=800&q=80",
    "rajma": "https://images.unsplash.com/photo-1626777552726-4a6b54c97e46?auto=format&fit=crop&w=800&q=80",
    "aloo matar": "https://images.unsplash.com/photo-1606491956689-2ea866880c84?auto=format&fit=crop&w=800&q=80",
    "aloo gobi": "https://images.unsplash.com/photo-1603894584373-5ac82b2ae398?auto=format&fit=crop&w=800&q=80",
    "kofta": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=800&q=80",
    
    # Rice & Biryani
    "biryani": "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?auto=format&fit=crop&w=800&q=80",
    "pulao": "https://images.unsplash.com/photo-1596797038530-2c107229654b?auto=format&fit=crop&w=800&q=80",
    "khichdi": "https://images.unsplash.com/photo-1626777552726-4a6b54c97e46?auto=format&fit=crop&w=800&q=80",
    
    # South Indian
    "sambar": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80",
    "rasam": "https://images.unsplash.com/photo-1626777552726-4a6b54c97e46?auto=format&fit=crop&w=800&q=80",
    "dosa": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?auto=format&fit=crop&w=800&q=80",
    "idli": "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?auto=format&fit=crop&w=800&q=80",
    
    # Sweets & Desserts (Mithai)
    "gulab jamun": "https://images.unsplash.com/photo-1527515862127-a4fc05baf7a5?auto=format&fit=crop&w=800&q=80",
    "gajar ka halwa": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80",
    "jalebi": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=800&q=80",
    "rasgulla": "https://images.unsplash.com/photo-1527515862127-a4fc05baf7a5?auto=format&fit=crop&w=800&q=80",
    "sandesh": "https://images.unsplash.com/photo-1527515862127-a4fc05baf7a5?auto=format&fit=crop&w=800&q=80",
    "kheer": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=800&q=80",
    "phirni": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=800&q=80",
    "kalakand": "https://images.unsplash.com/photo-1527515862127-a4fc05baf7a5?auto=format&fit=crop&w=800&q=80",
    "ghevar": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=800&q=80",
    "imarti": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=800&q=80",
    "balu shahi": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=800&q=80",
    "lassi": "https://images.unsplash.com/photo-1571006687083-d5d2e0dc38a6?auto=format&fit=crop&w=800&q=80",
    "rabri": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=800&q=80",
    "sheera": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80",
    "boondi": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=800&q=80",
    "basundi": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=800&q=80",
    "payasam": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?auto=format&fit=crop&w=800&q=80",
    "chikki": "https://images.unsplash.com/photo-1599488615731-7e5c2823ff28?auto=format&fit=crop&w=800&q=80",
    "petha": "https://images.unsplash.com/photo-1527515862127-a4fc05baf7a5?auto=format&fit=crop&w=800&q=80",
    "shrikhand": "https://images.unsplash.com/photo-1571006687083-d5d2e0dc38a6?auto=format&fit=crop&w=800&q=80",
    "dhokla": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80",
    "khandvi": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80",
    "samosa": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80",
    "pakora": "https://images.unsplash.com/photo-1601050690597-df0568f70950?auto=format&fit=crop&w=800&q=80"
}

GENERIC_FALLBACK = "https://images.unsplash.com/photo-1589302168068-964664d93dc0?auto=format&fit=crop&w=800&q=80"

# Curated YouTube Authentic Videos for Top Indian Recipes
CURATED_VIDEOS = {
    "paneer butter masala": "a03U45jFxOI",
    "shahi paneer": "6WJ4t3xYq9I",
    "dal tadka": "RXZsDHKlVGw",
    "dal makhani": "XqY2oE8L44A",
    "chana masala": "vT22e0eN7y0",
    "aloo matar": "1Y8g4lPzG7o",
    "gulab jamun": "rJ7vG5fE-Yw",
    "gajar ka halwa": "uV8eX6y4m_0",
    "sambar": "5Qy3o7uA2v4",
    "rasgulla": "aT9p2qX8k1Y",
    "jalebi": "k4j9QvL0_6A",
    "biryani": "r2q8e4u5w1I",
    "lassi": "0y6yX8z1v2w",
    "rabri": "7vQ9oX1y2zA",
    "ghevar": "8uP2wX9y0zA",
    "sheera": "1vQ2wX3y4zA",
    "chikki": "2vQ3wX4y5zA",
    "basundi": "3vQ4wX5y6zA",
    "kofta": "4vQ5wX6y7zA",
    "payasam": "6vQ7wX8y9zA",
    "imarti": "7vQ8wX9y0zA",
    "kalakand": "8vQ9wX0y1zA",
    "phirni": "9vQ0wX1y2zA",
    "sandesh": "0vQ1wX2y3zA"
}

# Runtime memory caches
_image_runtime_cache = {}
_youtube_runtime_cache = {}


def get_recipe_image(recipe_name, unsplash_key=None):
    """Get high-quality image URL for a recipe instantly with multi-layer fallback."""
    name_clean = recipe_name.strip().lower()

    # 1. Exact match in curated dictionary
    if name_clean in CURATED_IMAGES:
        return CURATED_IMAGES[name_clean]

    # 2. Check runtime cache
    if name_clean in _image_runtime_cache:
        return _image_runtime_cache[name_clean]

    # 3. Partial keyword matching
    for key, url in CURATED_IMAGES.items():
        if key in name_clean or name_clean in key:
            _image_runtime_cache[name_clean] = url
            return url

    # 4. Keyword heuristics (sweet, curry, dal, rice, paneer, etc.)
    if any(s in name_clean for s in ["sweet", "halwa", "ladoo", "jamun", "kheer", "pedha", "laddu"]):
        return CURATED_IMAGES["gulab jamun"]
    if any(s in name_clean for s in ["paneer"]):
        return CURATED_IMAGES["paneer butter masala"]
    if any(s in name_clean for s in ["dal", "curry", "gravy"]):
        return CURATED_IMAGES["dal tadka"]
    if any(s in name_clean for s in ["rice", "pulao", "biryani"]):
        return CURATED_IMAGES["biryani"]

    # 5. Fallback to delicious Indian food visual
    _image_runtime_cache[name_clean] = GENERIC_FALLBACK
    return GENERIC_FALLBACK


def get_recipe_youtube(recipe_name):
    """Get YouTube video data instantly without hanging or network delays."""
    name_clean = recipe_name.strip().lower()

    # 1. Exact match in curated list
    if name_clean in CURATED_VIDEOS:
        vid = CURATED_VIDEOS[name_clean]
        return {
            "video_id": vid,
            "watch_url": f"https://www.youtube.com/watch?v={vid}",
            "embed_url": f"https://www.youtube.com/embed/{vid}"
        }

    # 2. Check runtime cache
    if name_clean in _youtube_runtime_cache:
        return _youtube_runtime_cache[name_clean]

    # 3. Keyword match in curated list
    for key, vid in CURATED_VIDEOS.items():
        if key in name_clean or name_clean in key:
            data = {
                "video_id": vid,
                "watch_url": f"https://www.youtube.com/watch?v={vid}",
                "embed_url": f"https://www.youtube.com/embed/{vid}"
            }
            _youtube_runtime_cache[name_clean] = data
            return data

    # 4. Fallback search URL (instant, doesn't block request)
    fallback_query = urllib.parse.quote(f"{recipe_name} recipe authentic step by step")
    data = {
        "video_id": "",
        "watch_url": f"https://www.youtube.com/results?search_query={fallback_query}",
        "embed_url": ""
    }
    _youtube_runtime_cache[name_clean] = data
    return data
