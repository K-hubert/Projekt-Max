import json
import streamlit as st

@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def build_ingredient_lookup(data: dict):
    lookup = {}
    synonyms = []
    for item in data.get('skladniki', []):
        kcal = item.get('kalorie_na_100g', 0)
        std_w = item.get('waga_standardowa', 100)
        # nazwa
        name = item.get('nazwa', '').strip().lower()
        if name:
            lookup[name] = (kcal, std_w)
            synonyms.append(name)
        # synonimy
        for syn in item.get('synonimy', []):
            s = syn.strip().lower()
            lookup[s] = (kcal, std_w)
            synonyms.append(s)
    return lookup, sorted(set(synonyms))


def calculate_calories(ingredients_str: str, lookup: dict) -> int:
    total = 0
    for part in ingredients_str.split(','):
        ing = part.strip().lower()
        if ing in lookup:
            kcal100, w = lookup[ing]
            total += kcal100 * w / 100
    return round(total)


def get_available_recipes(user_ings: list, recipes: list, diet: str) -> list:
    matched = []
    for r in recipes:
        cat = r.get('category', '')
        ings = [i.strip().lower() for i in r.get('ingredients','').split(',')]
        if (diet=='dowolna' or cat==diet) and all(i in user_ings for i in ings):
            matched.append(r)
    return matched


def get_missing_ingredients(user_ings: list, recipes: list, diet: str) -> list:
    suggestions = []
    for r in recipes:
        cat = r.get('category','')
        ings = [i.strip().lower() for i in r.get('ingredients','').split(',')]
        if diet!='dowolna' and cat!=diet:
            continue
        missing = [i for i in ings if i not in user_ings]
        if 0 < len(missing) <=2:
            suggestions.append((r, missing))
    return suggestions