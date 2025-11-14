News Research Tool

A Streamlit-based AI app that allows users to paste news article URLs, extract content, generate embeddings, store them in FAISS, and ask questions and get accurate answers

🚀 Features

🔗 Load and scrape content from multiple news article URLs

✂️ Split text into chunks using LangChain

🔍 Generate embeddings using HuggingFace MiniLM L6 v2

📦 Store vectors in FAISS

🤖 Ask questions using Groq Llama 3.3 (70B)

📚 Retrieve answers with source references

🌐 Simple and interactive Streamlit UI

🛠️ Tech Stack

Python

Streamlit

LangChain

Groq LLM (ChatGroq API)

FAISS

HuggingFace Embeddings

📂 Project Structure
news-research-tool/
│
├── app.py
├── faiss_store_hf.pkl   # Auto-generated after processing URLs
├── README.md
├── requirements.txt
└── .env

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/news-research-tool.git
cd news-research-tool

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate  # Mac / Linux
venv\Scripts\activate     # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file in your project root:

GROQ_API_KEY=your_groq_api_key_here




▶️ Run the App
streamlit run app.py

💡 How It Works

Enter up to 3 news URLs

App loads and processes articles

Text is chunked and embeddings are generated

FAISS index is saved for reuse

Ask any question from the processed articles

App generates an AI answer + sources
