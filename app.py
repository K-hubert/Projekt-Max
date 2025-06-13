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

# --- Load data ---

recipes = load_json("data/recipes.json")  
ingredients_data = load_json("data/ingredients.json")  # <- pełny słownik 

# ingredients_data is dict with key "skladniki"
ingredients_list = ingredients_data.get("skladniki", [])

# Build lookup for calories and synonyms
ing_lookup, all_ingredients = build_ingredient_lookup(ingredients_data)

# Initialize vector store for similarity search (RAG)
embed_model, embed_index = init_vector_store(recipes)

# --- Helper to retrieve context for RAG ---
def retrieve_context(user_msg):
    user_msg = normalize(user_msg)
    context = ""
    seen = set()
    for r in recipes:
        for ing in _ingredient_iter(r):
            if ing in user_msg and r['title'] not in seen:
                context += f"Przepis: {r['title']}\nSkładniki: {r['ingredients']}\nKategoria: {r['category']}\n\n"
                seen.add(r['title'])
                if len(seen) >= 5:
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
    """
    Generate short step-by-step instructions for one recipe using AI.
    """
    prompt = [
        {"role": "system", "content": "Jesteś asystentem kulinarnym podającym zwięzłe instrukcje."},
        {"role": "user", "content": (
            f"Przepis: '{title}'. Składniki: {ingredients}. "
            "Podaj krótką instrukcję przygotowania krok po kroku w punktach."
        )}
    ]
    return ai_chat(prompt)

#--- Instruction generation for mealplan(7 days)
def generate_instructions_bulk_cached(recipes_list):
    """
    Generate instructions for multiple recipes with caching, batched call to AI.
    recipes_list: list of dicts with keys "title" and "ingredients"
    Returns: list of instructions in the same order
    """
    if 'instructions_cache' not in st.session_state:
        st.session_state.instructions_cache = {}

    cached_results = []
    to_request = []
    indices_to_request = []

    for i, r in enumerate(recipes_list):
        key = f"{r['title']}||{r['ingredients']}"
        if key in st.session_state.instructions_cache:
            cached_results.append((i, st.session_state.instructions_cache[key]))
        else:
            to_request.append(r)
            indices_to_request.append(i)

    new_results = []
    if to_request:
        prompt_text = "Jesteś asystentem kulinarnym podającym zwięzłe instrukcje przygotowania przepisów.\n"
        prompt_text += "Dla każdego przepisu podaj krótką instrukcję krok po kroku w punktach.\n\n"
        for i, r in enumerate(to_request, 1):
            prompt_text += f"{i}. Przepis: '{r['title']}'. Składniki: {r['ingredients']}.\n"

        prompt = [
            {"role": "system", "content": "Jesteś asystentem kulinarnym podającym zwięzłe instrukcje."},
            {"role": "user", "content": prompt_text}
        ]

        response = ai_chat(prompt)
        # Split by numbered list "1.", "2.", etc.
        instructions = re.split(r"\n\d+\.\s", "\n" + response)
        instructions = [instr.strip() for instr in instructions if instr.strip()]

        # Pad or trim instructions to match requested recipes count
        while len(instructions) < len(to_request):
            instructions.append("Brak instrukcji — wygeneruj ręcznie.")
        if len(instructions) > len(to_request):
            instructions = instructions[:len(to_request)]

        for idx, instr in zip(indices_to_request, instructions):
            key = f"{recipes_list[idx]['title']}||{recipes_list[idx]['ingredients']}"
            st.session_state.instructions_cache[key] = instr
            new_results.append((idx, instr))

    all_results = cached_results + new_results
    all_results.sort(key=lambda x: x[0])

    return [instr for _, instr in all_results]


# --- Calories filtering for mealplan ---
def filter_meals_by_kcal(meals, kcal_min, kcal_max, meals_per_day):
    """
    Filter meals so that daily kcal sum fits in kcal_min..kcal_max.
    Simple heuristic: each meal kcal <= max kcal per meal (kcal_max/meals_per_day).
    """
    max_kcal_per_meal = kcal_max // meals_per_day
    filtered = [m for m in meals if m['kcal'] <= max_kcal_per_meal]
    return filtered


# --- UI Layout ---

st.title("🍲 Twój kuchenny asystent AI 2.0")

tabs = st.tabs(["🔍 Przepisy", "💬 Chat", "📅 Jadłospis"])

recipes_tab, chat_tab, mealplan_tab = tabs

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

# --- Chat tab ---
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

# --- Mealplan tab ---
with mealplan_tab:
    st.header("📅 Twój tygodniowy jadłospis AI")

    diet_plan = st.selectbox(
        "Wybierz dietę:",
        ["dowolna", "wege", "keto", "niskotłuszczowa", "niskocukrowa"],
        key="diet_mealplan"
    )

    have_plan = st.multiselect("Jakie składniki masz pod ręką?", options=all_ingredients, key="have_mealplan")

    meals_per_day = st.selectbox("Ile posiłków chcesz dziennie?", [1, 2, 3], index=2)

    kcal_min, kcal_max = st.slider(
        "Wybierz zakres kalorii na dzień:",
        min_value=800, max_value=4000, value=(1500, 2500), step=100, key="kcal_range"
    )

    if st.button("🔄 Generuj jadłospis"):
        if not have_plan:
            st.warning("Wybierz przynajmniej jeden składnik.")
        else:
            # Get exact meals (RAG)
            available_meals = get_available_recipes(have_plan, recipes, diet_plan)
            # Get fallback meals with missing ingredients
            fallback_meals = get_missing_ingredients(have_plan, recipes, diet_plan)

            combined_meals = available_meals + [r for r, _ in fallback_meals]

            # Calculate kcal for each meal
            for m in combined_meals:
                m['kcal'] = calculate_calories(m['ingredients'], ing_lookup)

            filtered_meals = filter_meals_by_kcal(combined_meals, kcal_min, kcal_max, meals_per_day)
            if not filtered_meals:
                st.warning("Brak przepisów mieszczących się w podanym zakresie kalorii.")
            else:
                st.success(f"Znaleziono {len(filtered_meals)} posiłków mieszczących się w zakresie kalorii.")
                # mealplan for 7 days * meals_per_day
                plan = []
                idx = 0
                total_needed = 7 * meals_per_day
                while len(plan) < total_needed:
                    plan.append(filtered_meals[idx % len(filtered_meals)])
                    idx += 1

                instructions_list = generate_instructions_bulk_cached(plan)

                
                # show mealplan
                for day in range(7):
                    with st.expander(f"🗓️ Dzień {day + 1}"):
                        for meal_num in range(meals_per_day):
                            i = day * meals_per_day + meal_num
                            r = plan[i]
                            title = r['title']
                            category = r.get('category', 'brak kategorii')
                            kcal = r.get('kcal', '–')
                            ingredients_str = r.get('ingredients', '')
                            ingredients_list = [i.strip() for i in ingredients_str.split(',')] if ingredients_str else []
                            instructions = instructions_list[i]

                            st.markdown(f"### 🍽️ {meal_num + 1}. {title} — {kcal} kcal ({category})")
                            st.markdown("**🧂 Składniki:**")
                            st.write(", ".join(ingredients_list) if ingredients_list else "Brak składników.")

                            st.markdown("**👨‍🍳 Instrukcje:**")
                            if not instructions or "brak" in instructions.lower() or "[błąd generowania]" in instructions.lower():
                                st.warning("Brak instrukcji — wygeneruj ręcznie.")
                            else:
                                st.markdown(instructions)

                            st.markdown("---")  # separator




