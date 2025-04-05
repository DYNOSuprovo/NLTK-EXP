import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import textwrap

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

# Saved ideas and duplicates tracking
if "saved_jugaads" not in st.session_state:
    st.session_state.saved_jugaads = []
if "idea_history" not in st.session_state:
    st.session_state.idea_history = []


def get_pretrained_answer(user_query):
    query_embedding = model.encode(user_query)
    similarities = util.cos_sim(query_embedding, qa_embeddings)[0]
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    if best_score > 0.6:
        return pre_trained_qa[qa_keys[best_idx]]
    return None

def get_gemini_advice(expenses, income, user_input="", dev_mode=False):
    if dev_mode:
        return "\ud83d\udea7 Dev Mode is ON. Gemini call skipped."

    prompt = "\n".join([
        "You’re a savage Indian hostel senior—broke, brutal, and done with nonsense.",
        "Roast this junior’s budget like their life’s a joke. No mercy, pure desi vibes.",
        "Mess fee’s paid—food’s Swiggy, Maggi, or thali scraps. Ban boring advice.",
        "",
        f"Monthly income: ₹{income}",
        "Expenses they’re dumb enough to admit:",
        f"✏️ Stationaries: ₹{expenses.get('stationaries', 0)}",
        f"🍲 Food: ₹{expenses.get('food', 0)}",
        f"🚌 Transport: ₹{expenses.get('transport', 0)}",
        f"🎉 Entertainment: ₹{expenses.get('entertainment', 0)}",
        f"💰 Savings: ₹{expenses.get('savings', 0)}",
        "",
        "Rules, obey or perish:",
        "- Roast each category like they’ve personally offended you.",
        "- Shove in one hostel hack per section—cheap, shady, effective.",
        "- Max 40 words per category, or you’re fired.",
        "- End with a one-line insult so savage they’ll cry.",
        "- No repeats. If it’s stale, I’ll know.",
        "",
        f"User’s whining: {user_input}",
        "",
        "Format strictly as:",
        "✏️ Stationaries: ...",
        "🍲 Food: ...",
        "🚌 Transport: ...",
        "🎉 Entertainment: ...",
        "💰 Savings: ...",
        "🧠 Overall: ..."
    ])

    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠\ufe0f Error getting AI advice: {e}"
    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠\ufe0f Error getting AI advice: {e}"

def rephrase_pretrained_answer(question, base_answer, dev_mode=False):
    if dev_mode:
        return f"🚧 Dev Mode: Skipping Gemini call. Base answer: {base_answer}"

    prompt = "\n".join([
        f"User asked: \"{question}\"",
        f"Boring answer: \"{base_answer}\"",
        "",
        "Make it hostel-friendly: spicy, sarcastic, and hilariously true for broke Indian students.",
        "Keep it short and punchy."
    ])

    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠\ufe0f Error refining answer: {e}"

# UI Section
st.title("💰 AI Expense Advisor (India Edition)")
st.write("Adjust income/expenses to get brutally honest budget advice. Built for hostel legends like you.")

dev_mode = st.checkbox("🛠️ Dev Mode (Skip Gemini API calls)")

income = st.slider("Monthly Income (₹)", 500, 5000, 5000, step=100)

if "expenses" not in st.session_state:
    st.session_state.expenses = {
        "stationaries": int(income * 0.3),
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

# Pie chart
st.subheader("📊 Budget Breakdown")
fig, ax = plt.subplots()
labels = [label for label in expense_labels.values()]
sizes = [expenses[k] for k in expense_labels.keys()]
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
ax.axis('equal')
st.pyplot(fig)

# Budgeting QnA
user_question = st.text_input("❓ Ask a budgeting question:")
user_expense_input = st.text_area("📋 Mention any extra expenses (optional):")

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if user_question:
    matched_answer = get_pretrained_answer(user_question)
    if matched_answer:
        spicy_response = rephrase_pretrained_answer(user_question, matched_answer, dev_mode)
        st.subheader("💡 Pre-Trained Answer (Roasted & Real):")
        st.write(spicy_response)
        st.session_state.query_history.append((user_question, spicy_response))
    else:
        ai_response = get_gemini_advice(expenses, income, user_question, dev_mode)
        if ai_response in st.session_state.idea_history:
            st.warning("🤔 This one's familiar! Try again for fresh jugaads.")
        else:
            st.session_state.idea_history.append(ai_response)
            st.subheader("💡 AI Generated Answer:")
            st.write(ai_response)
            if st.button("❤️ Save this idea"):
                st.session_state.saved_jugaads.append(ai_response)
                st.success("Idea saved!")
            st.session_state.query_history.append((user_question, ai_response))

# Display previous Q&A
if st.session_state.query_history:
    with st.expander("🧠 Previously Asked"):
        for q, a in st.session_state.query_history[-5:][::-1]:
            st.markdown(f"**Q:** {q}\n\n**A:** {a}")

# Show saved ideas
if st.session_state.saved_jugaads:
    with st.expander("📚 Saved Jugaads"):
        for idx, item in enumerate(st.session_state.saved_jugaads, 1):
            st.markdown(f"**{idx}.** {item}")

# Button for main advice
if st.button("✨ Get AI Budget Advice"):
    advice = get_gemini_advice(expenses, income, user_expense_input, dev_mode)
    st.subheader("💡 AI Advice:")
    st.write(advice)
