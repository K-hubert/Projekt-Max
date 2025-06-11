import streamlit as st
import re
from openai import OpenAI
from functools import lru_cache
from utils import (
    load_json,
    build_ingredient_lookup,
    calculate_calories,
    get_available_recipes,
    get_missing_ingredients
)
from vector_store import init_vector_store, query_similar_recipes
from utils import _ingredient_iter, normalize

# --- Configuration & Secrets ---
st.set_page_config(
    page_title="Kuchenny RAG 2.0", 
    page_icon="🍽️", 
    layout="wide"
)
API_KEY = st.secrets.get("OPENROUTER_API_KEY")
if not API_KEY:
    st.error("Brak klucza API. Dodaj OPENROUTER_API_KEY do sekcji secrets.toml.")

# --- Load and cache data ---
recipes = load_json("data/recipes.json")
ingredients_data = load_json("data/ingredients.json")
ing_lookup, all_ingredients = build_ingredient_lookup(ingredients_data)

# --- Initialize vector store ---
embed_model, embed_index = init_vector_store(recipes)

# --- Context retrieval for chat ---
def retrieve_context(user_msg):
    user_msg = normalize(user_msg)
    context = ""
    seen = set()
    for r in recipes:
        for ing in _ingredient_iter(r):
            if ing in user_msg and r['title'] not in seen:
                context += f"Przepis: {r['title']}\nSkładniki: {r['ingredients']}\nKategoria: {r['category']}\n\n"
                seen.add(r['title'])
                break
        if len(seen) >= 5:
            break
    return context.strip()

# --- OpenAI chat helper ---
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
def ai_chat(messages: list, user_msg: str = "") -> str:
    try:
        context = retrieve_context(user_msg)
        if context:
            messages = messages.copy()
            messages.insert(1, {
                "role": "system",
                "content": (
                    "Oto kilka przepisów, które mogą pomóc użytkownikowi:\n" +
                    context +
                    "\nW odpowiedzi wykorzystaj powyższe dane, jeśli pasują."
                )
            })
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-maverick:free",
            messages=messages
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Błąd API: {e}")
        return "[błąd generowania]"

# --- Instruction generation with caching ---
@lru_cache(maxsize=64)
def generate_instructions(title: str, ingredients: str) -> str:
    prompt = [
        {"role": "system", "content": "Jesteś asystentem kulinarnym podającym zwięzłe instrukcje."},
        {"role": "user", "content": (
            f"Przepis: '{title}'. Składniki: {ingredients}. "
            "Podaj krótką instrukcję przygotowania krok po kroku w punktach."
        )}
    ]
    return ai_chat(prompt)

# --- UI Layout ---
st.title("🍲 Twój kuchenny asystent AI 2.0")
recipes_tab, chat_tab = st.tabs(["🔍 Przepisy", "💬 Chat"])

def clean_unicode(text):
    return re.sub(r'[^\x00-\xFFąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s\w.,:;!?()\'\"%-]', '', text)

# --- Przepisy Tab ---
with recipes_tab:
    st.sidebar.header("Ustawienia przepisów")
    diet = st.sidebar.selectbox(
        "Wybierz dietę", 
        ["dowolna", "wege", "keto", "niskotłuszczowa", "niskocukrowa"]
    )
    have = st.sidebar.multiselect("Składniki, które masz", options=all_ingredients)
    
    if st.sidebar.button("Szukaj przepisów"):
        if not have:
            st.sidebar.warning("Wybierz przynajmniej jeden składnik.")
        else:
            exact = get_available_recipes(have, recipes, diet)
            if exact:
                st.subheader("Przepisy pasujące do Twoich składników")
                seen = set()
                for r in exact:
                    title = re.sub(r"\s+\d+$", "", r['title'])
                    if title in seen:
                        continue
                    seen.add(title)
                    kcal = calculate_calories(r['ingredients'], ing_lookup)
                    with st.expander(f"{title} — {r['category']} ({kcal} kcal)"):
                        st.write(f"**Składniki:** {r['ingredients']}")
                        instr = generate_instructions(title, r['ingredients'])
                        st.markdown(instr)
            else:
                st.warning("Brak dokładnych dopasowań.")
            partial = get_missing_ingredients(have, recipes, diet)
            if partial:
                st.subheader("Przepisy z brakującymi składnikami")
                seen = set()
                for r, missing in partial:
                    title = re.sub(r"\s+\d+$", "", r['title'])
                    if title in seen:
                        continue
                    seen.add(title)
                    kcal = calculate_calories(r['ingredients'], ing_lookup)
                    with st.expander(f"{title} — {r['category']} ({kcal} kcal)"):
                        st.write(f"**Składniki:** {r['ingredients']}")
                        st.write(f"**Brakuje:** {', '.join(missing)}")
                        instr = generate_instructions(title, r['ingredients'])
                        st.markdown(instr)
            st.divider()
            st.subheader("Sugestie AI")
            sims = query_similar_recipes(
                query=", ".join(have),
                recipes=recipes,
                model=embed_model,
                index=embed_index,
                top_k=10
            )
            seen_cat = set()
            count = 0
            for r in sims:
                cat = r['category']
                if cat in seen_cat:
                    continue
                seen_cat.add(cat)
                title = re.sub(r"\s+\d+$", "", r['title'])
                kcal = calculate_calories(r['ingredients'], ing_lookup)
                with st.expander(f"{title} — {cat} ({kcal} kcal)"):
                    st.write(f"**Składniki:** {r['ingredients']}")
                    instr = generate_instructions(title, r['ingredients'])
                    st.markdown(instr)
                count += 1
                if count >= 3:
                    break

# --- Chat Tab ---
with chat_tab:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": (
                "Jesteś pomocnym kuchennym asystentem AI. "
                "Odpowiadaj po Polsku i używaj sensownych słów. "
                "Unikaj dziwnych znaków, emotek i alfabetów innych niż łaciński."
                "Zawsze podawaj informacje o kaloriach w przepisach."
            )}
        ]
    user_msg = st.chat_input("Napisz do asystenta...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        reply = ai_chat(st.session_state.chat_history, user_msg=user_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    history = st.session_state.chat_history[1:]
    for msg in reversed(history):
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
