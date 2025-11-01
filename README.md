# NextHorizon

**NextHorizon** is an AI‑powered Streamlit application designed to provide intelligent insights, analytics, and agentic workflows. This project includes both frontend and backend logic, integrated with the OpenAI API and other data sources.

---

## 🚀 Features

* Streamlit‑based interactive UI
* Modular architecture (utils, ui, ai, config)
* Integrations with OpenAI, DuckDuckGo Search, and PDF/DOCX parsers
* Environment variable configuration using `.env`
* Ready for containerization with Docker

---

## 🧩 Project Structure

```
NextHorizon/
│
├── app.py                # Main Streamlit app entry point
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build definition
├── .dockerignore         # Docker build exclusions
├── .gitignore            # Git exclusions
├── utils/                # Helper modules and utilities
├── ui/                   # Streamlit UI components
├── ai/                   # AI‑related logic and API integrations
├── config/               # Configuration files and secrets
└── assets/               # Images, CSS, or static content
```

---

## ⚙️ Local Setup

### 1️⃣ Create and activate virtual environment

```bash
python3 -m venv nh
source nh/bin/activate   # On Windows: nh\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment variables

Create a `.env` file in the project root with keys such as:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 4️⃣ Run Streamlit app locally

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Setup

### 1️⃣ Build Docker image

```bash
docker build -t nexthorizon:latest .
```

### 2️⃣ Run container

```bash
docker run -d -p 8501:8501 --name nexthorizon_app nexthorizon:latest
```

### 3️⃣ View app

Visit: [http://localhost:8501](http://localhost:8501)

---

## 📦 Deployment Options

* **Azure App Service:** For quick web hosting of the Streamlit app.
* **Azure Kubernetes Service (AKS):** For scalable, container‑based deployment.
* **Docker Compose:** For local multi‑service setup if you add databases or APIs later.

---

## 🧠 Development Notes

* Use `.env` for API keys and environment configs.
* Keep the repo clean using `.gitignore` and `.dockerignore`.
* Update `requirements.txt` whenever new libraries are installed.
* Follow modular naming conventions (`utils/`, `ui/`, `ai/`).

---

## 🧾 License

This project is distributed under the **MIT License**.
