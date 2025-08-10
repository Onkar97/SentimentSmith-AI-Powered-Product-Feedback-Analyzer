# SentimentSmith – AI-Powered Product Feedback Analyzer

SentimentSmith ingests customer reviews, performs **NLP-based sentiment analysis**, and exposes insights via a **Flask** REST API. It includes database artifacts and scripts for storing results and powering a simple dashboard.

## Features
- **Data Ingestion** from CSV/files into a relational database
- **Preprocessing** (tokenization, vectorization) and **model training** for sentiment classification
- **REST API** (Flask) to serve sentiment scores and key product metrics
- **SQL Artifacts** (`Database.sql`) to create required tables and indices

## Tech Stack
- **Language:** Python 3.9+
- **Frameworks/Libraries:** Flask, scikit-learn, numpy, pandas (typical)
- **Database:** SQL (see `Database.sql`)

## Setup
```bash
# (Optional) create venv
python -m venv .venv
.\.venv\Scriptsctivate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

# Set environment variables as needed (DB connection, API keys)
# e.g., export DATABASE_URL=postgresql://user:pass@localhost:5432/sentiments
# or use a .env file
python app.py
```
App runs at: `http://localhost:5000` by default.

## Project Structure
```
app.py
requirements.txt
Database.sql
templates/            # if present - Flask templates
static/               # if present - CSS/JS for simple UI
```
## API (example)
- `POST /predict` – returns sentiment label/score for input text
- `GET /metrics` – aggregated product metrics
