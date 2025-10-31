# 🗣️ DADS – Stutter Detection App

**DADS** (Detecting and Analyzing Disfluencies in Speech) is an AI-powered application designed to detect stuttering in speech recordings.  
This app allows users to record or upload audio, analyze it, and visualize stuttering patterns efficiently.

**📂 Project Structure**

```
DADS/
│-- Scripts/             # Data Generation,Preprocessing,Utilities
│-- Model/               # Model training & Evaluation
|-- dataset/             # Extract the dataset here
│-- App/                 # Application Code (UI,API,Deployment)
```

[SEP28k-dataset (from kaggle)](https://www.kaggle.com/datasets/ikrbasak/sep-28k)
---

## 🚀 How to Use the App

### Step 1: Clone the Repository
- Run the following command in your terminal (replace < repo-link > with your actual repository link):
```bash
git clone <repo-link>
```

---
### Step 2: Create a Virtual Environment
- On macOS / Linux:

```bash
python3 -m venv .venv
```
- On Windows:

```bash
python -m venv .venv
```
---
### Step 3: Activate the Virtual Environment and Install Dependencies
Activate the environment:

- macOS / Linux:
```bash
source .venv/bin/activate
```
- Windows:
```bash
.venv\Scripts\activate
```
Install the required packages:
```bash
pip install -r requirements.txt
```
---
### Step 4: Run the App

Start the application by running:
- macOS / Linux
```bash
python3 App/run_app.py
```
- Windows
```bash
python App/run_app.py
```
---
### 📝 Commit Guidelines

Follow this convention:

```
[TYPE] Commit message

```

**Common types**:

- **[FIX]** – Bug fixes
- **[ADD]** – New features
- **[DOCS]** – Documentation changes
- **[MNT]** – Code refactoring & Maintenance
- **[TEST]** – Tests additions or fixes
