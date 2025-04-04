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

# Updated FAQ database for hostel students with mess + kitchen
pre_trained_qa = {
    "how to save on groceries": "Since your mess fees are prepaid, only buy groceries if you're planning to use the common kitchen. Buy staples in bulk and split costs with roommates to save more.",
    "how much should i save monthly": "As a college student, saving even ₹300–₹1000/month is a great start. Focus on building habits, not large numbers.",
    "how to reduce electricity bill": "Use shared kitchen appliances efficiently. Switch off lights/fans when leaving. Avoid overusing personal electronics.",
    "best way to track expenses": "Use basic apps like Walnut or Google Sheets. Track your daily UPI spends and weekly food/kitchen recharges.",
    "how to reduce transportation cost": "Walk or cycle within campus. For trips outside, use metro passes or split cab fares with friends.",
    "how to build an emergency fund": "Try saving ₹100–₹500/month aside. Use a separate UPI wallet for this so you don't touch it easily.",
    "should i invest or save": "Start by saving small, then learn basic investing through apps like Groww or Zerodha Varsity once you’ve built a small reserve.",
    "how to save money as a student": "Utilize mess food as much as possible. Use your kitchen smartly only when needed. Avoid unnecessary online spending.",
    "tips for saving money in india": "Minimize Swiggy/Zomato, avoid impulse UPI spends, and split OTT or WiFi bills with friends.",
    "how to avoid impulse buying": "Uninstall shopping apps, and follow a ₹500/month cap on non-essential spends.",
    "how to plan a monthly budget": "Split income into food/kitchen, transport, essentials (like SIM), and a small buffer for savings/emergencies.",
    "how to save for travel": "Cut back slightly on weekly junk food or movie outings, and move that amount to a travel wallet.",
    "what is a good savings goal": "Even ₹5000 saved over 6 months is a win. Don’t aim big—aim consistent.",
    "how to cut dining expenses": "Limit eating out. Use mess, or coordinate group cooking in the kitchen to share grocery costs.",
    "how to save on phone bills": "Switch to student-friendly plans (e.g., Vi Hero, Jio 299). Use college WiFi where possible.",
    "how to manage credit card bills": "If you have a card, set monthly UPI limit alerts and repay in full before the due date.",
    "how to split rent with roommates": "Use apps like Splitwise to track all shared expenses: rent, groceries, electricity, WiFi.",
    "how to avoid overspending": "Withdraw weekly cash and avoid UPI overuse. Stick to a spending limit for each category.",
    "how to reduce monthly subscriptions": "Use shared Spotify/Netflix family plans or drop OTT subscriptions if not used.",
    "how to manage money on low income": "Prioritize mess (already paid), essentials, and set ₹100/week aside if possible. Group expenses wherever possible."
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
You are an AI budget advisor for a group of Indian college students living in hostels. 

They have already paid for a fixed mess plan (food), but the food quality varies, so they sometimes cook using a shared kitchen (with free induction but they buy utensils and ingredients themselves). 

Other shared expenses include data plans, group trips, groceries, online subscriptions, and local travel. 

Monthly income is ₹{income}, and current self-reported expenses are: {expenses}.

{user_input}

Provide **realistic**, student-friendly budgeting tips that maintain comfort and flexibility, with suggestions on saving without cutting all fun.
"""
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error getting AI advice: {e}"

# Gemini AI for refining pre-trained answers
def rephrase_pretrained_answer(question, base_answer):
    prompt = f"""
A user asked: "{question}"
Here's a basic answer: "{base_answer}"

Rephrase it to be more practical, detailed, and easy to follow for an Indian hostel student. 
Make it friendly and clear.
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error refining answer: {e}"

# ─────────────────────────────────────────────
# Streamlit UI with Real-time Rebalance
# ─────────────────────────────────────────────

st.title("💰 AI Expense Advisor (India Hostel Edition)")
st.caption("👨‍🎓 Tailored for college students with prepaid mess + shared kitchen lifestyle")

income = st.slider("Monthly Income (₹)", 500, 5000, 5000)

# Initialize expenses
if "expenses" not in st.session_state:
    st.session_state.expenses = {
        "kitchen+groceries": int(income * 0.25),
        "data+wifi": int(income * 0.1),
        "transport": int(income * 0.15),
        "entertainment": int(income * 0.1),
        "savings": int(income * 0.4)
    }

# Rebalance all to fit within income before rendering
total = sum(st.session_state.expenses.values())
if total > income:
    overflow = total - income
    for k in st.session_state.expenses:
        prop = st.session_state.expenses[k] / total
        st.session_state.expenses[k] = max(0, int(st.session_state.expenses[k] - prop * overflow))

# Define labels
expense_labels = {
    "kitchen+groceries": "🍳 Kitchen & Groceries (₹)",
    "data+wifi": "📱 Data/WiFi/Phone (₹)",
    "transport": "🚌 Local Transport (₹)",
    "entertainment": "🎉 Chill/Streaming/Outings (₹)",
    "savings": "💰 Savings/Backup (₹)"
}

# Real-time rebalance sliders
for key, label in expense_labels.items():
    new_val = st.slider(label, 0, income, st.session_state.expenses[key], key=key)
    if new_val != st.session_state.expenses[key]:
        delta = new_val - st.session_state.expenses[key]
        st.session_state.expenses[key] = new_val

        # Rebalance others
        others = [k for k in st.session_state.expenses if k != key]
        total_other = sum([st.session_state.expenses[k] for k in others])
        for k in others:
            if total_other > 0:
                prop = st.session_state.expenses[k] / total_other
                st.session_state.expenses[k] = max(0, int(st.session_state.expenses[k] - prop * delta))

        st.experimental_rerun()

expenses = st.session_state.expenses

# Inputs
user_question = st.text_input("Ask a budgeting question:")
user_expense_input = st.text_area("Other group-based expense notes (optional)")

# FAQ / Custom QnA
if user_question:
    matched_answer = get_pretrained_answer(user_question)
    st.subheader("💡 AI Refined Answer:")
    if matched_answer:
        st.write(rephrase_pretrained_answer(user_question, matched_answer))
    else:
        st.write(get_gemini_advice(expenses, income, user_question))

# Budget Summary
if st.button("Get Budget Advice Summary"):
    st.subheader("📊 Personalized Budget Tips")
    st.write(get_gemini_advice(expenses, income, user_expense_input))
