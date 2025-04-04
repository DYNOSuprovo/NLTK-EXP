import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

# Configure API key
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API key is missing! Please check your .env file or Streamlit secrets.")

# Predefined FAQs
pre_trained_qa = {
    "how to reduce electricity bill": "Fan > AC. Unplug stuff. Free sunlight exists — use it.",
    "best way to track expenses": "Use Splitwise for friends, Google Sheets for peace of mind. Track chai runs too.",
    "how to reduce transportation cost": "Cycle, walk, or just bunk. Autos are not budget-friendly for broke students.",
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

def get_gemini_advice(expenses, income, user_input=""):
    prompt = f"""
You're a broke-but-wise Indian hostel senior advising junior students on budgeting.
They’ve already paid the mess fee (so yes, "food" here = Swiggy escapes, Maggi runs, or squad-cooked hostel thalis).

Monthly income: ₹{income}
Here’s their spending:

🏠 Rent: ₹{expenses.get("rent", 0)}
🍲 Food: ₹{expenses.get("food", 0)}
🚌 Transport: ₹{expenses.get("transport", 0)}
🎉 Entertainment: ₹{expenses.get("entertainment", 0)}
💰 Savings: ₹{expenses.get("savings", 0)}

Your job?
- Roast or respect each category.
- Suggest if it's too high, too low, or just right.
- Give sarcastic, practical, and hostel-life hacks.

Extra info from user: {user_input}

Respond ONLY in this format:
🏠 Rent: ...
🍲 Food: ...
🚌 Transport: ...
🎉 Entertainment: ...
💰 Savings: ...
🧠 Overall: ...
"""
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error getting AI advice: {e}"

def rephrase_pretrained_answer(question, base_answer):
    prompt = f"""
A user asked: "{question}"
Here’s a basic answer: "{base_answer}"

Make it spicy, sarcastic, and specific for broke Indian hostel students who live on Swiggy, chai, and hope.
Keep it short, real, and funny.
"""
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error refining answer: {e}"

# UI Section

st.title("💰 AI Expense Advisor (India Edition)")
st.write("Adjust income/expenses to get brutally honest budget advice. Built for hostel legends like you.")

income = st.slider("Monthly Income (₹)", 500, 5000, 5000, step=100)

if "expenses" not in st.session_state:
    st.session_state.expenses = {
        "rent": int(income * 0.3),
        "food": int(income * 0.25),
        "transport": int(income * 0.15),
        "entertainment": int(income * 0.1),
        "savings": int(income * 0.2)
    }

# Rebalance if total exceeds income
total = sum(st.session_state.expenses.values())
overflow = total - income
if overflow > 0:
    for k in st.session_state.expenses:
        st.session_state.expenses[k] -= int((st.session_state.expenses[k] / total) * overflow)
        st.session_state.expenses[k] = max(0, st.session_state.expenses[k])

expense_labels = {
    "rent": "🏠 Rent (₹)",
    "food": "🍲 Food (₹)",
    "transport": "🚌 Transport (₹)",
    "entertainment": "🎉 Entertainment (₹)",
    "savings": "💰 Savings (₹)"
}

rerun_needed = False
changed_key = None

for key, label in expense_labels.items():
    current_value = st.session_state.expenses[key]
    new_value = st.slider(label, 0, income, current_value, key=key)
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
user_question = st.text_input("❓ Ask a budgeting question:")
user_expense_input = st.text_area("📋 Mention any extra expenses (optional):")

if user_question:
    matched_answer = get_pretrained_answer(user_question)
    if matched_answer:
        st.subheader("💡 Pre-Trained Answer (Roasted & Real):")
        st.write(rephrase_pretrained_answer(user_question, matched_answer))
    else:
        st.subheader("💡 AI Generated Answer:")
        st.write(get_gemini_advice(expenses, income, user_question))

if st.button("✨ Get AI Budget Advice"):
    advice = get_gemini_advice(expenses, income, user_expense_input)
    st.subheader("💡 AI Advice:")
    st.write(advice)
