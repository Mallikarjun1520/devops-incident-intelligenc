# Final Setup (Windows PowerShell)

Requires Python 3.10 - 3.12 and an internet connection for the first run (model download).

```powershell
# 1. From the extracted project folder
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1
#    If PowerShell blocks the script, run this once and activate again:
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install dependencies (installs PyTorch, so it takes a few minutes)
pip install -r requirements.txt

# 4. Create your .env and add your Groq API key
Copy-Item .env.example .env
notepad .env          # set GROQ_API_KEY and save

# 5. Load the 400 historical incidents into ChromaDB (downloads the embedding model on first run)
python scripts/ingest_incidents.py

# 6. (Optional) run the tests - no API key needed
python -m pytest

# 7. Start the server
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>, paste an incident (or click **Upload Incident** and choose
`data/sample_incident.txt`) and click **Analyze Incident**.

Next time, you only need steps 2 and 7 (or double-click `run.bat`).

## Optional: VS Code Live Server

Keep Uvicorn running, then right-click `frontend/index.html` and select **Open with Live Server**.
The frontend automatically connects to the local API at `http://127.0.0.1:8000`. If Uvicorn uses a
different address, add `?api=http://127.0.0.1:8001` to the Live Server page URL.
