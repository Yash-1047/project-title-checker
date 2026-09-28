# 🤖 AI Based Project Similarity Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-306998?style=for-the-badge)](https://www.nltk.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

An intelligent web-based platform designed for academic institutions to eliminate duplicate and semantically similar project title submissions. Built using **Python, Flask, Machine Learning (TF-IDF & Cosine Similarity), NLP (NLTK), SQLite, and Modern Responsive UI**.

---

## 📌 Problem Statement

In universities and colleges, students frequently submit project titles that are identical or cosmetically rephrased versions of previously submitted titles (e.g., *"AI Attendance System using Face Recognition"* vs. *"Face Recognition based Smart Attendance App"*). 

Manual verification by faculty is time-consuming and error-prone. This system automates duplication detection using an **ensemble AI/ML and NLP similarity pipeline**, ensuring high originality and academic rigor.

---

## 🧠 AI/ML & NLP Similarity Pipeline

The system uses a multi-stage Natural Language Processing and Machine Learning pipeline:

```
[Raw Title Input]
       │
       ▼
 1. Text Normalization (Lowercasing & Special Characters Removal)
       │
       ▼
 2. NLTK Tokenization (Splitting text into linguistic tokens)
       │
       ▼
 3. Stopword Filtering (Standard English + Academic/Domain filler removal)
       │
       ▼
 4. Porter Stemming (Root word reduction: 'detecting' -> 'detect')
       │
       ▼
 5. Multi-Method Ensemble Scoring
       ├── TF-IDF + Cosine Similarity (Weight: 45%) — Measures semantic vector angle
       ├── Jaccard Similarity (Weight: 30%)        — Measures unique keyword set overlap
       └── Sequence Matcher (Weight: 25%)          — Measures character/token structural flow
       │
       ▼
 6. Threshold-based Evaluation:
       ├── Exact Match (100%)       ──► 🚫 REJECTED
       ├── Combined Score >= 55%    ──► 🚫 REJECTED (Shows existing title & % match)
       └── Combined Score < 55%     ──► ✅ ACCEPTED & SAVED
```

---

## 🎯 Key Features

### 👨‍🎓 Student Dashboard
- **Roll Number Authentication**: Secure student sign-up and login using college Roll Number.
- **Real-Time Similarity Check**: When submitting a title, the AI checks it against all existing approved/submitted titles.
- **Duplicate Prevention**: Rejects duplicate or semantically identical titles and displays the conflicting title and exact similarity percentage.
- **Faculty Warning Notifications**: Real-time alerts when faculty requests changes to a project title.
- **Title Revision**: Enables students to update their title (which undergoes re-verification by the AI).

### 🎓 Faculty Dashboard
- **Analytics Overview**: Live metrics for Total Students, Submitted Projects, Approved Titles, and Warned Titles.
- **AI-Monitored Projects Table**: Displays all submitted titles with Roll Numbers, student names, and AI-computed similarity scores.
- **Student Warning System**: Faculty can send direct warning feedback with custom messages instructing students to modify their titles.
- **One-Click Approval**: Faculty can approve titles that meet originality standards.
- **Search & Filter**: Real-time filtering by student name, roll number, or project title.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | Python 3, Flask | RESTful APIs, Session Management, Controller logic |
| **Machine Learning** | scikit-learn, NumPy | TF-IDF Vectorizer (N-grams), Cosine Similarity Matrix |
| **NLP** | NLTK (Natural Language Toolkit) | Word Tokenization, Stopwords Removal, Porter Stemmer |
| **Database** | SQLite3 | Relational storage with foreign key constraints |
| **Frontend** | HTML5, Modern CSS3, JavaScript (Fetch API) | Clean, responsive UI with zero external bulky frameworks |

---

## 📂 Project Directory Structure

```
project-title-checker/
│
├── app.py                 # Flask application entry point & API routes
├── ml_engine.py           # Core AI/ML & NLP Similarity Pipeline
├── requirements.txt       # Python dependencies list
├── .gitignore             # Git ignored files & directories
├── README.md              # Project documentation & evaluation guide
│
└── public/                # Static assets & frontend templates
    ├── login.html         # Roll No authentication & registration page
    ├── student.html       # Student project submission & status dashboard
    ├── faculty.html       # Faculty management, warnings & approvals dashboard
    └── css/
        └── style.css      # Clean, modern responsive design stylesheet
```

---

## 🚀 Setup & Installation (Judges / Evaluators)

### 1. Clone or Download the Repository
```bash
git clone <repository-url>
cd project-title-checker
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access the Web Application
Open your browser and navigate to:
```
http://localhost:3000
```

---

## 🔑 Demo & Testing Credentials

| Role | Roll No / ID | Password | Access Dashboard |
|---|---|---|---|
| **Faculty** | `faculty` | `faculty123` | [http://localhost:3000/faculty](http://localhost:3000/faculty) |
| **Student** | *Register any Roll No (e.g. `21CS001`)* | *Any password* | [http://localhost:3000/student](http://localhost:3000/student) |

---

## 🧪 Test Cases for Judges Evaluation

Follow these quick steps to evaluate the system in under 2 minutes:

### Test Case 1: Initial Submission (Approved Case)
1. Go to `http://localhost:3000` and register as Student 1:
   - **Roll No**: `21CS001`, **Name**: `Alice`, **Password**: `pass123`
2. Submit title:  
   `"AI Based Attendance System using Face Recognition"`
3. **Result**: ✅ **Accepted & Submitted successfully**.

### Test Case 2: Semantic Duplicate Detection (Rejection Case)
1. Open an Incognito window, register as Student 2:
   - **Roll No**: `21CS002`, **Name**: `Bob`, **Password**: `pass123`
2. Attempt to submit a semantically similar title:  
   `"Smart Attendance Management System Using Face Detection"`
3. **Result**: 🚫 **Rejected!** System identifies high similarity (~70%) with Alice's title and prevents submission.

### Test Case 3: Unique Title Submission (Acceptance Case)
1. As Student 2, enter a completely different title:  
   `"Blockchain Based Land Registry and Verification System"`
2. **Result**: ✅ **Accepted & Submitted successfully**.

### Test Case 4: Faculty Warning & Approval Workflow
1. Log in as Faculty:  
   - **ID**: `faculty`, **Password**: `faculty123`
2. Review the submitted titles in the dashboard table.
3. Click **⚠️ Warn** on any student's project and enter a note (e.g., *"Refine scope to include mobile app"*).
4. Switch back to the student's tab — the student receives the warning alert in real time and can edit their title.
5. In the faculty dashboard, click **✅ Approve** to finalize any project title.
