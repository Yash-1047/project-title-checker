# AI-Based Project Title Similarity Detection System

A simple MCA Information Retrieval project. It lets a user search project titles, compare relevance, find matching passages, and explore recommendations and other IR demonstrations.

## Run the project (Windows / VS Code)

1. Extract this ZIP.
2. In VS Code, open the folder that contains `app.py` and `requirements.txt`. The files are inside the `project-title-checker-main` folder.
3. Open Terminal → New Terminal and run:

```powershell
python -m pip install -r requirements.txt
python app.py
```

4. Open `http://localhost:3000` in Chrome.

## Login

- Faculty username: `faculty`
- Faculty password: `faculty123`
- To test as a student, register from the login page with a new roll number and password.

## Sample project records

On the first run, the app adds eight clearly labelled demo project records so Search & Ranking and Recommendations have data to work with. These are for testing only. They are not real student submissions. The app keeps them in `data.db`. If you already have a `data.db` with projects, it will not add another set.

Example searches:
- `Student Performance Prediction`
- `Face Recognition Attendance`
- `Movie Recommendation System`

## Main sections

- **Search & Ranking:** searches project titles using the selected ranking method.
- **Passage Retrieval:** ranks sentences from text you provide.
- **Recommendations:** suggests titles related to a query.
- **PageRank:** demonstrates link-based ranking on the project graph.
- **Evaluation:** evaluates rankings when relevance labels have been saved.
- **Web Crawler:** extracts readable text from a permitted public webpage.
- **Extraction & Semantic:** extracts keywords and related text information.
- **Parallel / MapReduce:** demonstrates parallel ranking and term counting locally.

Some sections need user-provided text, saved ratings, relevance labels, or public URLs to produce results.
