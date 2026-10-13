"""Flask blueprint exposing the Information Retrieval APIs (/api/ir/*)."""
from flask import Blueprint, jsonify, request, session

from ir import automaton, extraction, graph, parallel, ranking, recommend
from ir.crawler import crawl_page, parse_html

MAX_QUERY, MAX_TEXT = 500, 100_000

IR_SCHEMA = '''
CREATE TABLE IF NOT EXISTS project_ratings (
    user_id INTEGER NOT NULL, project_id INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, project_id),
    FOREIGN KEY (user_id) REFERENCES users(id), FOREIGN KEY (project_id) REFERENCES projects(id)
);
CREATE TABLE IF NOT EXISTS relevance_judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, project_id INTEGER NOT NULL,
    created_by INTEGER NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (query, project_id),
    FOREIGN KEY (project_id) REFERENCES projects(id), FOREIGN KEY (created_by) REFERENCES users(id)
);
'''


def ensure_ir_tables(conn):
    conn.executescript(IR_SCHEMA)
    conn.commit()


def create_ir_blueprint(get_db, require_login, require_student, require_faculty):
    bp = Blueprint("ir", __name__, url_prefix="/api/ir")

    def body():
        return request.get_json(silent=True) or {}

    def text_field(data, name, required=True, limit=MAX_QUERY):
        v = str(data.get(name, "")).strip()
        if required and not v:
            raise ValueError("'%s' is required." % name)
        if len(v) > limit:
            raise ValueError("'%s' is too long (max %d characters)." % (name, limit))
        return v

    def load_items(status=None):
        conn = get_db()
        try:
            rows = conn.execute("SELECT id, title, status FROM projects ORDER BY id").fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def method_of(data):
        m = str(data.get("method", "tfidf")).lower()
        if m not in ranking.METHODS:
            raise ValueError("Method must be one of: " + ", ".join(ranking.METHODS))
        return m

    @bp.errorhandler(ValueError)
    def bad_request(exc):
        return jsonify({"error": str(exc)}), 400

    @bp.route("/projects")
    @require_login
    def projects():
        return jsonify({"projects": load_items()})

    # ---------------- search & ranking ----------------
    @bp.route("/search", methods=["POST"])
    @require_login
    def search():
        data = body()
        q = text_field(data, "query")
        items = load_items()
        if not items:
            return jsonify({"query": q, "results": [], "message": "No projects in the database yet."})
        if data.get("method") == "all":
            comp = parallel.rank_all_methods(q, items)
            return jsonify({"query": q, "method": "all", "comparison": comp})
        m = method_of(data)
        results = ranking.rank(q, items, m)
        resp = {"query": q, "method": m, "results": results}
        # legacy keys kept for backwards compatibility
        resp["language_model_ranking"] = [{"title": r["title"], "score": r["score"]}
                                          for r in ranking.rank(q, items, "lm")]
        resp["link_based_pagerank"] = [{"title": r["title"], "rank": r["pagerank"]}
                                       for r in graph.rank_projects(items)["results"]]
        return jsonify(resp)

    @bp.route("/passages", methods=["POST"])
    @require_login
    def passages():
        data = body()
        q = text_field(data, "query")
        text = text_field(data, "text", limit=MAX_TEXT)
        return jsonify({"results": ranking.rank_passages(q, text, method_of(data))})

    @bp.route("/semantic", methods=["POST"])
    @require_login
    def semantic():
        return jsonify(extraction.semantic_search(text_field(body(), "query"), load_items()))

    @bp.route("/pagerank", methods=["POST"])
    @require_login
    def pagerank():
        try:
            th = float(body().get("threshold", 0.15))
        except (TypeError, ValueError):
            raise ValueError("Threshold must be a number between 0 and 1.")
        if not 0 <= th <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        return jsonify(graph.rank_projects(load_items(), th))

    @bp.route("/automaton", methods=["POST"])
    @require_login
    def automaton_demo():
        data = body()
        terms = [t for t in str(data.get("terms", "")).replace(",", " ").split() if t][:20]
        text = text_field(data, "text", limit=5000)
        return jsonify(automaton.match(terms, text))

    # ---------------- parallel / mapreduce ----------------
    @bp.route("/mapreduce", methods=["POST"])
    @require_login
    def mapreduce():
        data = body()
        docs = data.get("documents")
        if docs is None:
            docs = [i["title"] for i in load_items()]
        if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
            raise ValueError("'documents' must be a list of strings.")
        if not docs:
            return jsonify({"message": "No documents available.", "terms": []})
        return jsonify(parallel.mapreduce_term_frequency(docs[:1000]))

    # ---------------- recommenders ----------------
    @bp.route("/recommendations", methods=["POST"])
    @require_login
    def recommendations():
        data = body()
        q = text_field(data, "query")
        if len(q) < 2:
            raise ValueError("Enter at least 2 characters to find recommendations.")
        exclude = data.get("project_id")
        recs = recommend.content_based(q, load_items(), 3, exclude if isinstance(exclude, int) else None)
        return jsonify({"method": "Content-Based Recommendation (TF-IDF + Cosine Similarity)",
                        "recommendations": [dict(r, title=r["title"], score=r["similarity_percent"])
                                            for r in recs]})

    @bp.route("/my-recommendations")
    @require_student
    def my_recommendations():
        items = load_items()
        conn = get_db()
        try:
            row = conn.execute("SELECT id FROM projects WHERE student_id = ?", (session["user_id"],)).fetchone()
        finally:
            conn.close()
        mine = next((i for i in items if row and i["id"] == row["id"]), None)
        if not mine:
            return jsonify({"recommendations": [], "message": "Submit a project title to get recommendations."})
        return jsonify({"project": mine["title"],
                        "recommendations": recommend.content_based(mine["title"], items, 3, mine["id"])})

    def _owner(project_id):
        conn = get_db()
        try:
            r = conn.execute("SELECT student_id FROM projects WHERE id = ?", (project_id,)).fetchone()
        finally:
            conn.close()
        return r["student_id"] if r else None

    @bp.route("/recommend/knowledge", methods=["POST"])
    @require_login
    def knowledge():
        data = body()
        status = str(data.get("status", "")).strip() or None
        if status not in (None, "submitted", "approved", "warned"):
            raise ValueError("Invalid status.")
        split = lambda k: [w for w in str(data.get(k, "")).replace(",", " ").split() if w][:10]
        try:
            min_words = max(0, min(int(data.get("min_words") or 0), 50))
        except (TypeError, ValueError):
            raise ValueError("min_words must be a number.")
        if not (status or split("include") or split("exclude") or min_words):
            raise ValueError("Choose at least one criterion.")
        return jsonify({"results": recommend.knowledge_based(load_items(), status, split("include"),
                                                             split("exclude"), min_words)})

    @bp.route("/ratings", methods=["POST"])
    @require_student
    def rate():
        data = body()
        try:
            pid, rating = int(data.get("project_id")), int(data.get("rating"))
        except (TypeError, ValueError):
            raise ValueError("project_id and rating are required.")
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5.")
        if _owner(pid) is None:
            return jsonify({"error": "Project not found."}), 404
        if _owner(pid) == session["user_id"]:
            raise ValueError("You cannot rate your own project.")
        conn = get_db()
        try:
            conn.execute("INSERT INTO project_ratings (user_id, project_id, rating) VALUES (?,?,?) "
                         "ON CONFLICT(user_id, project_id) DO UPDATE SET rating=excluded.rating",
                         (session["user_id"], pid, rating))
            conn.commit()
        finally:
            conn.close()
        return jsonify({"message": "Rating saved."})

    @bp.route("/recommend/collaborative")
    @require_student
    def collaborative():
        conn = get_db()
        try:
            rows = conn.execute("SELECT user_id, project_id, rating FROM project_ratings").fetchall()
        finally:
            conn.close()
        recs, message = recommend.collaborative([tuple(r) for r in rows], session["user_id"])
        titles = {i["id"]: i["title"] for i in load_items()}
        for r in recs:
            r["title"] = titles.get(r["project_id"], "")
        return jsonify({"recommendations": recs, "message": message, "total_ratings": len(rows)})

    # ---------------- evaluation ----------------
    @bp.route("/judgments", methods=["GET", "POST", "DELETE"])
    @require_login
    def judgments():
        conn = get_db()
        try:
            if request.method == "GET":
                rows = conn.execute("SELECT j.id, j.query, j.project_id, p.title FROM relevance_judgments j "
                                    "JOIN projects p ON p.id = j.project_id ORDER BY j.query, j.id").fetchall()
                return jsonify({"judgments": [dict(r) for r in rows]})
            if session.get("role") != "faculty":
                return jsonify({"error": "Access denied. Faculty only."}), 403
            data = body()
            if request.method == "DELETE":
                conn.execute("DELETE FROM relevance_judgments WHERE id = ?", (int(data.get("id", 0)),))
                conn.commit()
                return jsonify({"message": "Judgment removed."})
            q = text_field(data, "query")
            ids = data.get("project_ids")
            if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
                raise ValueError("Select at least one relevant project.")
            for pid in ids:
                if conn.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone():
                    conn.execute("INSERT OR IGNORE INTO relevance_judgments (query, project_id, created_by) "
                                 "VALUES (?,?,?)", (q, pid, session["user_id"]))
            conn.commit()
            return jsonify({"message": "Relevance labels saved."}), 201
        finally:
            conn.close()

    @bp.route("/evaluate", methods=["POST"])
    @require_login
    def evaluate():
        try:
            k = max(1, min(int(body().get("k", 3)), 50))
        except (TypeError, ValueError):
            raise ValueError("k must be a whole number.")
        conn = get_db()
        try:
            rows = conn.execute("SELECT query, project_id FROM relevance_judgments").fetchall()
        finally:
            conn.close()
        if not rows:
            return jsonify({"evaluations": [], "message": "No relevance labels saved yet. "
                            "Label relevant projects for a query first; no results are fabricated."})
        judg = {}
        for r in rows:
            judg.setdefault(r["query"], set()).add(r["project_id"])
        items = load_items()
        evals = [e for e in (ranking.evaluate(judg, items, m, k) for m in ranking.METHODS) if e]
        return jsonify({"evaluations": evals, "queries": len(judg)})

    # ---------------- extraction & crawling ----------------
    @bp.route("/extract", methods=["POST"])
    @require_login
    def extract():
        data = body()
        html = str(data.get("html", ""))
        text = str(data.get("text", ""))
        if not (html.strip() or text.strip()):
            raise ValueError("Provide HTML or text to extract.")
        if len(html) + len(text) > MAX_TEXT * 2:
            raise ValueError("Input is too large.")
        info = extraction.extract_all(text, html)
        plain = parse_html(html)["text"] if html.strip() else text
        return jsonify({"extracted_text": plain, "extraction": info})

    @bp.route("/project-extraction")
    @require_login
    def project_extraction():
        titles = ". ".join(i["title"] for i in load_items())
        return jsonify({"extraction": extraction.extract_all(titles)})

    @bp.route("/crawl", methods=["POST"])
    @require_faculty
    def crawl():
        url = text_field(body(), "url", limit=2048)
        try:
            result = crawl_page(url)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": "Could not crawl this page: " + type(exc).__name__}), 502
        result["extraction"] = extraction.extract_all(result["text"])
        return jsonify(result)

    return bp
