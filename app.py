"""
AI Based Project Similarity Detection System
=============================================
Flask Backend — Uses ML/NLP pipeline for intelligent title matching.

Tech Stack: Python, Flask, scikit-learn, NLTK, SQLite
"""

import os
import sqlite3
from datetime import timedelta
from functools import wraps

from flask import Flask, request, jsonify, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash

# Import our AI/ML engine
from ml_engine import engine as similarity_engine

# ============================================================
# App Configuration
# ============================================================

app = Flask(__name__, static_folder='public', static_url_path='')
app.secret_key = 'ai-project-similarity-detection-secret-2024'
app.permanent_session_lifetime = timedelta(hours=24)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.db')


# ============================================================
# Database Helpers
# ============================================================

def get_db():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables and seed default faculty account."""
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'faculty')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL UNIQUE,
            title TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'submitted' CHECK(status IN ('submitted', 'approved', 'warned')),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS warnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            faculty_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            is_read INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects(id),
            FOREIGN KEY (faculty_id) REFERENCES users(id)
        );
    ''')

    # Seed default faculty account
    existing = conn.execute(
        'SELECT id FROM users WHERE username = ?', ('faculty',)
    ).fetchone()
    if not existing:
        hashed = generate_password_hash('faculty123')
        conn.execute(
            'INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)',
            ('faculty', hashed, 'Dr. Faculty', 'faculty')
        )
        conn.commit()
        print('[OK] Default faculty account created: username=faculty, password=faculty123')

    conn.close()


# ============================================================
# Auth Decorators
# ============================================================

def require_student(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Please log in to continue'}), 401
        if session.get('role') != 'student':
            return jsonify({'error': 'Access denied. Students only.'}), 403
        return f(*args, **kwargs)
    return decorated


def require_faculty(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Please log in to continue'}), 401
        if session.get('role') != 'faculty':
            return jsonify({'error': 'Access denied. Faculty only.'}), 403
        return f(*args, **kwargs)
    return decorated


# ============================================================
# Auth Routes
# ============================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    name = data.get('name', '').strip()

    if not username or not password or not name:
        return jsonify({'error': 'All fields are required'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Roll No must be at least 3 characters'}), 400
    if len(password) < 4:
        return jsonify({'error': 'Password must be at least 4 characters'}), 400

    conn = get_db()
    if conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone():
        conn.close()
        return jsonify({'error': 'This Roll No is already registered'}), 409

    hashed = generate_password_hash(password)
    cursor = conn.execute(
        'INSERT INTO users (username, password, name, role) VALUES (?, ?, ?, ?)',
        (username, hashed, name, 'student')
    )
    conn.commit()
    uid = cursor.lastrowid
    conn.close()

    session.permanent = True
    session['user_id'] = uid
    session['username'] = username
    session['name'] = name
    session['role'] = 'student'

    return jsonify({
        'message': 'Registration successful',
        'user': {'id': uid, 'username': username, 'name': name, 'role': 'student'}
    }), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid username or password'}), 401

    session.permanent = True
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['name'] = user['name']
    session['role'] = user['role']

    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user['id'], 'username': user['username'],
            'name': user['name'], 'role': user['role']
        }
    })


@app.route('/api/auth/me')
def me():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    return jsonify({'user': {
        'id': session['user_id'], 'username': session['username'],
        'name': session['name'], 'role': session['role']
    }})


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})


# ============================================================
# Student Routes
# ============================================================

def _get_existing_titles(exclude_student_id=None):
    """Fetch all project titles, optionally excluding one student."""
    conn = get_db()
    if exclude_student_id:
        rows = conn.execute(
            'SELECT title FROM projects WHERE student_id != ?',
            (exclude_student_id,)
        ).fetchall()
    else:
        rows = conn.execute('SELECT title FROM projects').fetchall()
    conn.close()
    return [r['title'] for r in rows]


@app.route('/api/student/project')
@require_student
def get_student_project():
    conn = get_db()
    project = conn.execute('''
        SELECT p.*, u.name as student_name, u.username
        FROM projects p JOIN users u ON p.student_id = u.id
        WHERE p.student_id = ?
    ''', (session['user_id'],)).fetchone()
    conn.close()
    return jsonify({'project': dict(project) if project else None})


@app.route('/api/student/project', methods=['POST'])
@require_student
def submit_project():
    """Submit a new project title — runs through the AI/ML similarity pipeline."""
    data = request.get_json()
    title = data.get('title', '').strip()

    if not title or len(title) < 5:
        return jsonify({'error': 'Project title must be at least 5 characters long'}), 400

    conn = get_db()
    if conn.execute('SELECT id FROM projects WHERE student_id = ?', (session['user_id'],)).fetchone():
        conn.close()
        return jsonify({'error': 'You already have a project. Use update to change your title.'}), 409
    conn.close()

    # --- AI/ML Similarity Check ---
    existing = _get_existing_titles(exclude_student_id=session['user_id'])
    is_rejected, similar_titles, analysis = similarity_engine.check_title(title, existing)

    if is_rejected:
        exact = any(s.get('is_exact_match') for s in similar_titles)
        return jsonify({
            'error': ('This exact title already exists! Choose a different title.'
                      if exact
                      else 'AI detected your title is too similar to existing title(s). Choose a more unique title.'),
            'similarTitles': similar_titles,
            'analysis': analysis,
            'rejected': True,
        }), 409

    # Accepted — save
    conn = get_db()
    cur = conn.execute(
        'INSERT INTO projects (student_id, title, status) VALUES (?, ?, ?)',
        (session['user_id'], title, 'submitted')
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()

    return jsonify({
        'message': 'Project title accepted by AI! Submitted successfully.',
        'project': {'id': pid, 'title': title, 'status': 'submitted'},
        'analysis': analysis,
    }), 201


@app.route('/api/student/project', methods=['PUT'])
@require_student
def update_project():
    """Update project title — re-runs AI/ML similarity check."""
    data = request.get_json()
    title = data.get('title', '').strip()

    if not title or len(title) < 5:
        return jsonify({'error': 'Project title must be at least 5 characters long'}), 400

    conn = get_db()
    existing_proj = conn.execute(
        'SELECT id FROM projects WHERE student_id = ?', (session['user_id'],)
    ).fetchone()
    conn.close()

    if not existing_proj:
        return jsonify({'error': 'No project found. Submit a new title.'}), 404

    # --- AI/ML Similarity Check ---
    existing = _get_existing_titles(exclude_student_id=session['user_id'])
    is_rejected, similar_titles, analysis = similarity_engine.check_title(title, existing)

    if is_rejected:
        exact = any(s.get('is_exact_match') for s in similar_titles)
        return jsonify({
            'error': ('This exact title already exists! Choose a different title.'
                      if exact
                      else 'AI detected your title is too similar to existing title(s). Choose a more unique title.'),
            'similarTitles': similar_titles,
            'analysis': analysis,
            'rejected': True,
        }), 409

    conn = get_db()
    conn.execute(
        'UPDATE projects SET title = ?, status = ?, updated_at = CURRENT_TIMESTAMP WHERE student_id = ?',
        (title, 'submitted', session['user_id'])
    )
    conn.commit()
    conn.close()

    return jsonify({
        'message': 'Title updated and accepted by AI!',
        'project': {'id': existing_proj['id'], 'title': title, 'status': 'submitted'},
        'analysis': analysis,
    })


@app.route('/api/student/warnings')
@require_student
def get_student_warnings():
    conn = get_db()
    warnings = conn.execute('''
        SELECT w.*, u.name as faculty_name
        FROM warnings w
        JOIN users u ON w.faculty_id = u.id
        JOIN projects p ON w.project_id = p.id
        WHERE p.student_id = ?
        ORDER BY w.created_at DESC
    ''', (session['user_id'],)).fetchall()

    project = conn.execute(
        'SELECT id FROM projects WHERE student_id = ?', (session['user_id'],)
    ).fetchone()
    if project:
        conn.execute(
            'UPDATE warnings SET is_read = 1 WHERE project_id = ? AND is_read = 0',
            (project['id'],)
        )
        conn.commit()
    conn.close()
    return jsonify({'warnings': [dict(w) for w in warnings]})


@app.route('/api/student/warnings/count')
@require_student
def warning_count():
    conn = get_db()
    project = conn.execute(
        'SELECT id FROM projects WHERE student_id = ?', (session['user_id'],)
    ).fetchone()
    if not project:
        conn.close()
        return jsonify({'count': 0})
    r = conn.execute(
        'SELECT COUNT(*) as c FROM warnings WHERE project_id = ? AND is_read = 0',
        (project['id'],)
    ).fetchone()
    conn.close()
    return jsonify({'count': r['c']})


# ============================================================
# Faculty Routes
# ============================================================

@app.route('/api/faculty/projects')
@require_faculty
def faculty_projects():
    conn = get_db()
    projects = conn.execute('''
        SELECT p.*, u.name as student_name, u.username as student_username,
            (SELECT COUNT(*) FROM warnings w WHERE w.project_id = p.id) as warning_count
        FROM projects p
        JOIN users u ON p.student_id = u.id
        ORDER BY p.created_at DESC
    ''').fetchall()
    conn.close()

    projects_list = [dict(p) for p in projects]

    # AI/ML: compute pairwise similarities
    if len(projects_list) >= 2:
        titles = [p['title'] for p in projects_list]
        sim_map = similarity_engine.pairwise_max_similarity(titles)
        for p in projects_list:
            max_sim, most_similar = sim_map.get(p['title'], (0, ''))
            p['max_similarity'] = max_sim
            p['most_similar_to'] = most_similar
    else:
        for p in projects_list:
            p['max_similarity'] = 0
            p['most_similar_to'] = ''

    return jsonify({'projects': projects_list})


@app.route('/api/faculty/warn/<int:project_id>', methods=['POST'])
@require_faculty
def warn_student(project_id):
    data = request.get_json()
    message = data.get('message', '').strip()
    if not message or len(message) < 5:
        return jsonify({'error': 'Warning message must be at least 5 characters'}), 400

    conn = get_db()
    if not conn.execute('SELECT id FROM projects WHERE id = ?', (project_id,)).fetchone():
        conn.close()
        return jsonify({'error': 'Project not found'}), 404

    conn.execute(
        'INSERT INTO warnings (project_id, faculty_id, message) VALUES (?, ?, ?)',
        (project_id, session['user_id'], message)
    )
    conn.execute(
        'UPDATE projects SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
        ('warned', project_id)
    )
    conn.commit()
    conn.close()
    return jsonify({'message': 'Warning sent to student successfully'}), 201


@app.route('/api/faculty/approve/<int:project_id>', methods=['POST'])
@require_faculty
def approve_project(project_id):
    conn = get_db()
    if not conn.execute('SELECT id FROM projects WHERE id = ?', (project_id,)).fetchone():
        conn.close()
        return jsonify({'error': 'Project not found'}), 404
    conn.execute(
        'UPDATE projects SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?',
        ('approved', project_id)
    )
    conn.commit()
    conn.close()
    return jsonify({'message': 'Project title approved successfully'})


@app.route('/api/faculty/stats')
@require_faculty
def faculty_stats():
    conn = get_db()
    stats = {
        'totalStudents':    conn.execute("SELECT COUNT(*) as c FROM users WHERE role='student'").fetchone()['c'],
        'totalProjects':    conn.execute("SELECT COUNT(*) as c FROM projects").fetchone()['c'],
        'submittedProjects': conn.execute("SELECT COUNT(*) as c FROM projects WHERE status='submitted'").fetchone()['c'],
        'approvedProjects': conn.execute("SELECT COUNT(*) as c FROM projects WHERE status='approved'").fetchone()['c'],
        'warnedProjects':   conn.execute("SELECT COUNT(*) as c FROM projects WHERE status='warned'").fetchone()['c'],
        'totalWarnings':    conn.execute("SELECT COUNT(*) as c FROM warnings").fetchone()['c'],
    }
    conn.close()
    return jsonify({'stats': stats})


# ============================================================
# Page Routes
# ============================================================

@app.route('/')
def index():
    return send_from_directory('public', 'login.html')

@app.route('/student')
def student_page():
    return send_from_directory('public', 'student.html')

@app.route('/faculty')
def faculty_page():
    return send_from_directory('public', 'faculty.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('public', path)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 3000))

    print()
    print('=' * 52)
    print('  AI Based Project Similarity Detection System')
    print(f'  Running at: http://localhost:{port}')
    print('=' * 52)
    print()
    print('  Faculty Login:  username=faculty  password=faculty123')
    print()

    app.run(host='0.0.0.0', port=port, debug=True)
