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
    st.error("⚠️ API key is missing! Please check your .env file.")

# Pre-trained FAQ answers
pre_trained_qa = {
    "how to save on groceries": "Create a monthly grocery list, buy in bulk for staples, and track weekly spending to avoid overspending.",
    "how much should i save monthly": "Aim to save at least 20% of your monthly income. Adjust this based on fixed expenses and priorities.",
    "how to reduce electricity bill": "Use energy-efficient appliances, turn off devices when not in use, and monitor monthly power usage.",
    "best way to track expenses": "Use apps like Walnut or Excel sheets to log daily spending and review your monthly outflow.",
    "how to reduce transportation cost": "Plan commute routes in advance, use monthly public transport passes, or consider ride-sharing to cut costs.",
    "how to build an emergency fund": "Set aside a fixed monthly amount, like ₹500–₹2000, until you accumulate 3–6 months' worth of expenses.",
    "should i invest or save": "First cover monthly savings goals (e.g., rent, bills, emergency fund), then invest leftover funds wisely.",
    "how to save money as a student": "Track monthly expenses on food and transport, use campus amenities, and avoid frequent takeouts.",
    "tips for saving money in india": "Limit impulsive UPI payments, avoid mid-month overspending, and review your monthly expenses every weekend.",
    "how to avoid impulse buying": "Stick to a monthly budget for non-essentials and avoid online browsing without intent.",
    "how to plan a monthly budget": "Split income into needs (50%), wants (30%), and savings (20%). Adjust based on actual monthly expenses.",
    "how to save for travel": "Include a travel fund in your monthly budget—cut back slightly on luxuries like entertainment or food delivery.",
    "what is a good savings goal": "Saving 20–30% of monthly income is a healthy goal. Set clear targets like ₹10,000 in 6 months.",
    "how to cut dining expenses": "Limit dine-outs to once or twice a month. Track how much you spend on food apps monthly.",
    "how to save on phone bills": "Switch to a plan that fits your monthly usage. Track data usage to avoid overages.",
    "how to manage credit card bills": "Pay off the full due amount every month. Set a fixed credit limit for yourself to avoid overuse.",
    "how to split rent with roommates": "Use apps like Splitwise and ensure each person's share is included in monthly planning.",
    "how to avoid overspending": "Categorize monthly expenses, avoid frequent UPI app usage, and stick to weekly spending caps.",
    "how to reduce monthly subscriptions": "Audit all your subscriptions once a month. Cancel unused ones or switch to family plans.",
    "how to manage money on low income": "Track every rupee, prioritize essentials, and create a strict monthly budget with a fixed saving goal—even ₹100 helps."
}

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
qa_keys = list(pre_trained_qa.keys())
qa_embeddings = model.encode(qa_keys)

# Match user query to pre-trained
def get_pretrained_answer(user_query):
    query_embedding = model.encode(user_query)
    similarities = util.cos_sim(query_embedding, qa_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    if best_score > 0.6:
        return pre_trained_qa[qa_keys[best_idx]]
    return None

# Gemini AI for advice
def get_gemini_advice(expenses, income, user_input=""):
    prompt = f"""
    A group of Indian college students living in a hostel have already paid for their monthly mess.
    However, mess food quality is inconsistent — some days it's good, some days it's bad.
    They also have access to a shared kitchen with free induction stoves, but they must buy ingredients and utensils themselves.

    Their monthly income is ₹{income}. Here are the current estimated monthly expenses: {expenses}.
    {user_input}

    Suggest creative, practical ways they can save money, improve budgeting, and make smarter financial choices.
    Keep the tone friendly and focused on college life.
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error getting AI advice: {e}"

# Rephrase pre-trained answer
def rephrase_pretrained_answer(question, base_answer):
    prompt = f"""
    A user asked: "{question}"
    Here's a basic answer: "{base_answer}"
    
    Rewrite it with more helpful details and a friendly tone suitable for Indian college students managing hostel life and monthly expenses.
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error refining answer: {e}"

# ─────────────────────────────────────────────
# UI with Dynamic Expense Rebalancing
# ─────────────────────────────────────────────

st.title("💰 AI Expense Advisor (India Edition)")
st.write("Adjust your income and expenses to get budget advice. Built for hostel students who want smart money tips.")

income = st.slider("Monthly Income (₹)", 500, 5000, 5000, step=100)

# Initialize session state
if "expenses" not in st.session_state:
    st.session_state.expenses = {
        "rent": int(income * 0.3),
        "food": int(income * 0.25),
        "transport": int(income * 0.15),
        "entertainment": int(income * 0.1),
        "savings": int(income * 0.2)
    }

# Rebalance on total overflow
total = sum(st.session_state.expenses.values())
overflow = total - income

if overflow > 0:
    adjustable_keys = list(st.session_state.expenses.keys())
    for k in adjustable_keys:
        st.session_state.expenses[k] -= int((st.session_state.expenses[k] / total) * overflow)
        st.session_state.expenses[k] = max(0, st.session_state.expenses[k])

# Expense labels with emojis
expense_labels = {
    "rent": "🏠 Rent/Mortgage (₹)",
    "food": "🍲 Food Expenses (₹)",
    "transport": "🚌 Transport (₹)",
    "entertainment": "🎉 Entertainment (₹)",
    "savings": "💰 Savings (₹)"
}

# Draw sliders and detect changes
for key, label in expense_labels.items():
    new_value = st.slider(label, 0, income, st.session_state.expenses[key], key=key)

    if new_value != st.session_state.expenses[key]:
        delta = new_value - st.session_state.expenses[key]
        st.session_state.expenses[key] = new_value

        # Redistribute among others
        other_keys = [k for k in st.session_state.expenses if k != key]
        total_others = sum([st.session_state.expenses[k] for k in other_keys])

        for k in other_keys:
            if total_others > 0:
                proportion = st.session_state.expenses[k] / total_others
                st.session_state.expenses[k] = max(0, st.session_state.expenses[k] - int(proportion * delta))

        st.experimental_rerun()

# Inputs
expenses = st.session_state.expenses
user_question = st.text_input("❓ Ask a budgeting question:")
user_expense_input = st.text_area("📋 Mention any extra expenses (optional):")

# Handle question
if user_question:
    matched_answer = get_pretrained_answer(user_question)
    if matched_answer:
        st.subheader("💡 Pre-Trained Answer (Refined):")
        st.write(rephrase_pretrained_answer(user_question, matched_answer))
    else:
        st.subheader("💡 AI Generated Answer:")
        st.write(get_gemini_advice(expenses, income, user_question))

# Button for general advice
if st.button("✨ Get AI Budget Advice"):
    advice = get_gemini_advice(expenses, income, user_expense_input)
    st.subheader("💡 AI Advice:")
    st.write(advice)
