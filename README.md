# 💸 AI Expense Advisor (India Edition)

> Hostel broke? Salary gone in Maggi and chai? This AI advisor gives *real*, *sarcastic*, and *spot-on* budget advice just like that brutally honest hostel senior.

---

## 📆 Features

- 🔘 **Slider Input**: Set your monthly income and category-wise expenses.
- 🤖 **Gemini Integration**: Sarcastic advice generated using Google Gemini.
- 🧠 **FAQ Matching**: Preloaded questions matched using Sentence Transformers.
- 📊 **Pie Chart Breakdown**: Visualize where your money’s vanishing.
- 🔄 **Query History**: See your last 5 burns... I mean, questions.

---

## ✨ Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/ai-expense-advisor.git
cd ai-expense-advisor
```

### 2. Setup Environment

Make a `.env` file like this:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run your_script.py
```

> Replace `your_script.py` with your actual file name if it's not `main.py`.

---

## 🔧 Dev Mode

Enable the **🛠️ Dev Mode** checkbox in the UI to skip Gemini calls and simulate responses. Saves API calls while debugging.

---

## 📂 File Structure

```text
├── your_script.py        # Streamlit app
├── .env                  # API key
├── requirements.txt      # Dependencies
├── README.md             # You're reading it
```

---

## 🧪 Requirements

Create this file for dependency setup:

### `requirements.txt`

```txt
streamlit
google-generativeai
python-dotenv
sentence-transformers
matplotlib
```

---

## 📢 Note

This app is **tailored for Indian hostel students** — full of realistic advice and desi sarcasm. It’s **not** your average budgeting app. It’s better.

---

## 📜 License

MIT — reuse, remix, and roast your budget however you like.

---

> *“Because nothing teaches budgeting better than being broke before the 10th of the month.”*

