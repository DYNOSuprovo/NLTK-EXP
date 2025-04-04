import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import textwrap
import random

# Load environment variables
load_dotenv()

# Configure API key
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("\u26a0\ufe0f API key is missing! Please check your .env file or Streamlit secrets.")

# Predefined FAQs
pre_trained_qa = {
    "how to reduce electricity bill": "Fan > AC. Unplug stuff. Free sunlight exists — use it.",
    "best way to track expenses": "Use Splitwise for friends, Google Sheets for peace of mind. Track chai runs too.",
    "how to reduce transportation cost": "Cycle, walk, or just bunk. Autos are not budget-friendly for broke students."
}

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
qa_keys = list(pre_trained_qa.keys())
qa_embeddings = model.encode(qa_keys)

def get_pretrained_answer(user_query):
    query_embedding = model.encode(user_query)
    similarities = util.cos_sim(query_embedding, qa_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    if best_score > 0.6:
        return pre_trained_qa[qa_keys[best_idx]]
    return None

def random_intro():
    return random.choice([
        "Let's audit this mess of a budget, shall we?",
        "Okay, let’s see where your money is *evaporating*.",
        "Financial report card time! Spoiler: You might fail.",
        "Welcome to Budget Roast 101."
    ])

def random_outro():
    return random.choice([
        "In short, stop acting rich on a poor man’s budget.",
        "Hope this helps you survive till the next UPI alert.",
        "Now go, live like a legend — or at least a functioning broke student.",
        "At this rate, you’ll be eating Maggi for dinner. Every day. Forever."
    ])

def get_gemini_advice(expenses, income, user_input="", dev_mode=False):
    if dev_mode:
        return "\ud83d\udea7 Dev Mode is ON. Gemini call skipped."

    categories = list(expenses.keys())
    random.shuffle(categories)

    example_section = "\n".join([
        "Examples of tone:",
        "✏️ Stationaries: Buying 3 pens a month is fine. Buying 3 types of highlighters to color-code your already empty schedule? No.",
        "🍲 Food: ₹700 on Swiggy and ₹20 on groceries? Bro, that’s not budgeting, that’s betrayal of Maggi.",
        "🚌 Transport: If Uber knows your name, you’re not budgeting right."
    ])

    prompt = "\n".join([
        random_intro(),
        "You're a broke-but-wise Indian hostel senior advising junior students on budgeting.",
        "They’ve already paid the mess fee (so yes, 'food' here = Swiggy escapes, Maggi runs, or squad-cooked hostel thalis).",
        f"Monthly income: ₹{income}",
        "Here’s their spending:",
        *[f"{key.capitalize()}: ₹{expenses[key]}" for key in categories],
        "",
        example_section,
        "",
        "Your job:",
        "- Roast or respect each category.",
        "- Suggest if it's too high, too low, or just right.",
        "- Give sarcastic, practical, and hostel-life hacks.",
        f"Extra info from user: {user_input}",
        "",
        "Respond ONLY in this format:",
        *[f"{key.capitalize()}: ..." for key in categories],
        "🧠 Overall: ...",
        random_outro()
    ])

    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"\u26a0\ufe0f Error getting AI advice: {e}"

def rephrase_pretrained_answer(question, base_answer, dev_mode=False):
    if dev_mode:
        return f"\ud83d\udea7 Dev Mode: Skipping Gemini call. Base answer: {base_answer}"

    prompt = "\n".join([
        f"A user asked: \"{question}\"",
        f"Here’s a basic answer: \"{base_answer}\"",
        "",
        "Make it spicy, sarcastic, and specific for broke Indian hostel students who live on Swiggy, chai, and hope.",
        "Keep it short, real, and funny."
    ])

    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"\u26a0\ufe0f Error refining answer: {e}"

# UI Section
st.title("\ud83d\udcb0 AI Expense Advisor (India Edition)")
st.write("Adjust income/expenses to get brutally honest budget advice. Built for hostel legends like you.")

dev_mode = st.checkbox("\ud83d\udee0\ufe0f Dev Mode (Skip Gemini API calls)")

income = st.slider("Monthly Income (₹)", 500, 5000, 5000, step=100)

if "expenses" not in st.session_state:
    st.session_state.expenses = {
        "stationaries": int(income * 0.3),
        "food": int(income * 0.25),
        "transport": int(income * 0.15),
        "entertainment": int(income * 0.1),
        "savings": int(income * 0.2)
    }

total = sum(st.session_state.expenses.values())
overflow = total - income
if overflow > 0:
    for k in st.session_state.expenses:
        st.session_state.expenses[k] -= int((st.session_state.expenses[k] / total) * overflow)
        st.session_state.expenses[k] = max(0, st.session_state.expenses[k])

expense_labels = {
    "stationaries": "✏️ Stationaries (₹)",
    "food": "🍲 Food (₹)",
    "transport": "🚌 Transport (₹)",
    "entertainment": "🎉 Entertainment (₹)",
    "savings": "💰 Savings (₹)"
}

rerun_needed = False
changed_key = None

for key, label in expense_labels.items():
    current_value = st.session_state.expenses[key]
    new_value = st.slider(label, 0, income, current_value, step=10, key=key)
    if new_value != current_value and not rerun_needed:
        delta = new_value - current_value
        st.session_state.expenses[key] = new_value
        changed_key = key
        rerun_needed = True

if rerun_needed and changed_key:
    other_keys = [k for k in st.session_state.expenses if k != changed_key]
    total_others = sum([st.session_state.expenses[k] for k in other_keys])
    for k in other_keys:
        if total_others > 0:
            proportion = st.session_state.expenses[k] / total_others
            st.session_state.expenses[k] = max(0, st.session_state.expenses[k] - int(proportion * delta))
    st.rerun()

expenses = st.session_state.expenses

st.subheader("\ud83d\udcca Budget Breakdown")
fig, ax = plt.subplots()
labels = [label for label in expense_labels.values()]
sizes = [expenses[k] for k in expense_labels.keys()]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
ax.axis('equal')
st.pyplot(fig)

user_question = st.text_input("\u2753 Ask a budgeting question:")
user_expense_input = st.text_area("\ud83d\udccb Mention any extra expenses (optional):")

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if user_question:
    matched_answer = get_pretrained_answer(user_question)
    if matched_answer:
        spicy_response = rephrase_pretrained_answer(user_question, matched_answer, dev_mode)
        st.subheader("\ud83d\udca1 Pre-Trained Answer (Roasted & Real):")
        st.write(spicy_response)
        st.session_state.query_history.append((user_question, spicy_response))
    else:
        ai_response = get_gemini_advice(expenses, income, user_question, dev_mode)
        st.subheader("\ud83d\udca1 AI Generated Answer:")
        st.write(ai_response)
        st.session_state.query_history.append((user_question, ai_response))

if st.session_state.query_history:
    with st.expander("\ud83e\udde0 Previously Asked"):
        for q, a in st.session_state.query_history[-5:][::-1]:
            st.markdown(f"**Q:** {q}\n\n**A:** {a}")

if st.button("\u2728 Get AI Budget Advice"):
    advice = get_gemini_advice(expenses, income, user_expense_input, dev_mode)
    st.subheader("\ud83d\udca1 AI Advice:")
    st.write(advice)
