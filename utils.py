import json
import streamlit as st

def normalize(text: str) -> str:
    """proste czyszczenie – spacje, lower-case"""
    return " ".join(text.lower().split())

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def build_ingredient_lookup(data: dict):
    lookup = {}
    canon = []                       # ← tu przechowujemy tylko 'nazwa'

    for item in data.get("skladniki", []):
        kcal = item.get("kalorie_na_100g", 0)
        std = item.get("waga_standardowa", 100)

        # nazwa główna (singular)
        name = normalize(item.get("nazwa", ""))
        if name:
            lookup[name] = (kcal, std)
            canon.append(name)       # dodajemy TYLKO nazwę główną

        # wszystkie synonimy (mogą być w liczbie mnogiej itp.)
        for syn in item.get("synonimy", []):
            s = normalize(syn)
            lookup[s] = (kcal, std)  # w lookup zostają – potrzebne do przeliczeń

    return lookup, sorted(set(canon))  # ← kanoniczne, bez duplikatów

def _ingredient_iter(recipe: dict):
    """
    Zwraca listę nazw składników:
    - jeśli przepis ma 'ingredients_detail', bierzemy stamtąd
    - inaczej dzielimy field 'ingredients' po przecinku
    """
    if 'ingredients_detail' in recipe:
        return [normalize(i['item']) for i in recipe['ingredients_detail']]
    return [normalize(i) for i in recipe.get('ingredients', '').split(',') if i.strip()]

def calculate_calories(recipe_ings, lookup: dict) -> int:
    """
    Przyjmuje string lub listę dictów (zależnie od formatu w pliku JSON).
    """
    total = 0
    if isinstance(recipe_ings, str):
        parts = [normalize(p) for p in recipe_ings.split(',') if p.strip()]
        for ing in parts:
            if ing in lookup:
                kcal100, w = lookup[ing]
                total += kcal100 * w / 100
    elif isinstance(recipe_ings, list):  # ingredients_detail
        for row in recipe_ings:
            ing = normalize(row['item'])
            qty = row.get('quantity', 1)
            if ing in lookup:
                kcal100, w_std = lookup[ing]
                w = qty * w_std
                total += kcal100 * w / 100
    return round(total)

def get_available_recipes(user_ings: list, recipes: list, diet: str) -> list:
    user_set = {normalize(i) for i in user_ings}
    matched = []
    for r in recipes:
        cat = r.get('category', '')
        ings = _ingredient_iter(r)
        if (diet == 'dowolna' or cat == diet) and user_set.issuperset(ings):
            matched.append(r)
    return matched

def get_missing_ingredients(user_ings: list, recipes: list, diet: str) -> list:
    user_set = {normalize(i) for i in user_ings}
    suggestions = []
    for r in recipes:
        cat = r.get('category', '')
        if diet != 'dowolna' and cat != diet:
            continue
        ings = _ingredient_iter(r)
        missing = [i for i in ings if i not in user_set]
        if 0 < len(missing) <= 2:
            suggestions.append((r, missing))
    return suggestions
