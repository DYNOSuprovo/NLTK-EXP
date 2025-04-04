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
    My monthly income is ₹{income}. Here are my expenses: {expenses}.
    {user_input}
    Analyze my budget and suggest practical ways to save money without affecting my lifestyle.
    Provide specific, actionable tips.
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
    
    Please rewrite it to be more helpful, detailed, and easy to understand.
    Use a friendly and practical tone, suitable for someone new to personal finance.
    """
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error refining answer: {e}"

# Streamlit UI
st.title("💰 AI Expense Advisor (India Edition)")
st.write("Adjust your income and expenses to get budget advice.")

income = st.slider("Monthly Income (₹)", 500, 5000, 5000)
rent = st.slider("Rent/Mortgage (₹)", 0, 2000, 1500)
food = st.slider("Food Expenses (₹)", 0, 1500, 8000)
transport = st.slider("Transport (₹)", 0, 500, 1000)
entertainment = st.slider("Entertainment (₹)", 0, 500, 1000)
savings = st.slider("Savings (₹)", 0, 200, 100)

user_question = st.text_input("Ask a budgeting question:")
user_expense_input = st.text_area("Describe any other expenses (optional)")

expenses = {
    "rent": rent,
    "food": food,
    "transport": transport,
    "entertainment": entertainment,
    "savings": savings
}

# Process Question
if user_question:
    matched_answer = get_pretrained_answer(user_question)
    if matched_answer:
        st.subheader("💡 AI Refined Answer from Pre-trained:")
        st.write(rephrase_pretrained_answer(user_question, matched_answer))
    else:
        st.subheader("💡 AI Generated Answer:")
        st.write(get_gemini_advice(expenses, income, user_question))

# Budget Summary Advice
if st.button("Get AI Budget Advice"):
    advice = get_gemini_advice(expenses, income, user_expense_input)
    st.subheader("💡 AI Advice:")
    st.write(advice)
