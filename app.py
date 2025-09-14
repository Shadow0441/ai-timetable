import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from flask import (Flask, render_template, request, jsonify, session,
                   redirect, url_for, abort, flash)
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy import create_engine, text
from celery import Celery, Task
from stable_baselines3 import PPO

# Load environment variables from .env file
load_dotenv()

# Import ML engine components
from hackathon_timetable import (EnhancedTimetableDemo, HybridTimetableEnv, EnhancedConfig, train_and_save_model)
from nlp_parser import NLPParser

# --- Application Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory cache for database engines
engine_cache = {}

def make_celery(app):
    """Initializes a Celery object with the Flask app context."""
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)

    class ContextTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app = Flask(__name__, instance_relative_config=True)

# --- Configuration ---
app.config.from_mapping(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key'),
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///skylink.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    CELERY_BROKER_URL=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

# Ensure the instance folder exists for storing models
try:
    os.makedirs(os.path.join(app.instance_path, 'trained_models'))
except OSError:
    pass

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
celery = make_celery(app)


# =============================================================================
# DATABASE MODELS
# =============================================================================
class Institute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    db_url = db.Column(db.String(500), nullable=True)
    config_json = db.Column(db.Text, nullable=True, default='{}')

    model_trained = db.Column(db.Boolean, default=False, index=True)
    model_training_status = db.Column(db.String(50), default='pending')
    model_path = db.Column(db.String(500), nullable=True)
    last_training = db.Column(db.DateTime, nullable=True)

    generations = db.relationship('TimetableGeneration', backref='institute', lazy=True, cascade="all, delete-orphan")
    students = db.relationship('Student', backref='institute', lazy=True, cascade="all, delete-orphan")

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False, index=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(80), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('institute_id', 'username', name='_institute_username_uc'),)

class TimetableGeneration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False, index=True)
    timetable_data = db.Column(db.Text, nullable=True)
    generation_time = db.Column(db.DateTime, default=datetime.utcnow)
    task_completion_rate = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(50), default='pending', index=True)
    nlp_query = db.Column(db.Text, nullable=True)


# =============================================================================
# DATABASE ADAPTER
# =============================================================================
class DatabaseAdapter:
    """Handles connection and data extraction from an institute's external database."""
    def __init__(self, db_url: str):
        if not db_url:
            raise ValueError("Database URL for institute cannot be empty.")
        self.db_url = db_url
        self.engine = None

    def connect(self):
        """Connects to the database, using a cache to avoid recreating engines."""
        try:
            if self.db_url in engine_cache:
                self.engine = engine_cache[self.db_url]
            else:
                self.engine = create_engine(self.db_url, connect_args={'connect_timeout': 10})
                engine_cache[self.db_url] = self.engine

            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Institute database connection failed: {e}")
            return False

    def extract_data(self, schema_config: dict):
        """Extracts timetable data using a dynamic schema mapping."""
        if not self.engine:
            raise ConnectionError("Database engine not connected.")
        if not schema_config or not all(k in schema_config for k in ['courses', 'faculty', 'rooms', 'map']):
            raise ValueError("Schema configuration is missing or incomplete.")

        try:
            with self.engine.connect() as conn:
                courses_cfg = schema_config['courses']
                faculty_cfg = schema_config['faculty']
                rooms_cfg = schema_config['rooms']
                map_cfg = schema_config['map']

                courses = [r[0] for r in conn.execute(text(f"SELECT {courses_cfg['col_name']} FROM {courses_cfg['table']}"))]
                faculty = [r[0] for r in conn.execute(text(f"SELECT {faculty_cfg['col_name']} FROM {faculty_cfg['table']}"))]
                rooms = [r[0] for r in conn.execute(text(f"SELECT {rooms_cfg['col_name']} FROM {rooms_cfg['table']}"))]

                map_query = text(f"""
                    SELECT c.{courses_cfg['col_name']}, f.{faculty_cfg['col_name']}
                    FROM {map_cfg['table']}
                    JOIN {courses_cfg['table']} c ON {map_cfg['table']}.{map_cfg['col_course_fk']} = c.id
                    JOIN {faculty_cfg['table']} f ON {map_cfg['table']}.{map_cfg['col_faculty_fk']} = f.id
                """)
                course_faculty_map = {c_name: f_name for c_name, f_name in conn.execute(map_query)}

            if not all([courses, faculty, rooms, course_faculty_map]):
                raise ValueError("Data extraction returned empty lists. Check schema mapping and database content.")

            return courses, faculty, rooms, course_faculty_map, True
        except Exception as e:
            logger.error(f"Data extraction failed with custom schema: {e}", exc_info=True)
            return None, None, None, None, False


# =============================================================================
# CELERY BACKGROUND TASKS
# =============================================================================
def get_model_path_for_institute(username):
    """Generates a safe, unique file path for an institute's trained model."""
    safe_username = "".join(c for c in username if c.isalnum())
    return os.path.join(app.instance_path, 'trained_models', f'trained_model_{safe_username}.zip')

@celery.task(bind=True)
def train_custom_model(self, institute_id):
    """Trains a custom ML model based on the institute's own database."""
    logger.info(f"Starting custom model training for institute {institute_id}.")

    with app.app_context():
        institute = Institute.query.get(institute_id)
        if not institute:
            logger.error(f"Institute {institute_id} not found")
            return

        try:
            institute.model_training_status = 'training'
            db.session.commit()

            schema_config = json.loads(institute.config_json or '{}')

            # Try to extract real data if DB URL and config exist
            courses, faculty, rooms, map_data = None, None, None, None
            if institute.db_url and schema_config:
                try:
                    adapter = DatabaseAdapter(institute.db_url)
                    if adapter.connect():
                        courses, faculty, rooms, map_data, success = adapter.extract_data(schema_config)
                        if not success:
                            raise ValueError("Data extraction failed")
                except Exception as e:
                    logger.warning(f"Failed to extract real data for institute {institute_id}: {e}")

            # Use default data if extraction failed or no DB configured
            if not courses:
                logger.info(f"Using default data for institute {institute_id} training.")
                courses = ['Math', 'Physics', 'Chemistry', 'Biology', 'English']
                faculty = ['Dr. Smith', 'Prof. Johnson', 'Dr. Williams', 'Prof. Brown']
                rooms = ['Room A', 'Room B', 'Room C', 'Lab 1', 'Lab 2']
                map_data = {
                    'Math': 'Dr. Smith',
                    'Physics': 'Prof. Johnson',
                    'Chemistry': 'Dr. Williams',
                    'Biology': 'Prof. Brown',
                    'English': 'Dr. Smith'
                }

            model_path = get_model_path_for_institute(institute.username)
            train_and_save_model(courses, faculty, rooms, map_data, save_path=model_path)

            institute.model_path = model_path
            institute.model_trained = True
            institute.last_training = datetime.utcnow()
            institute.model_training_status = 'completed'
            db.session.commit()
            logger.info(f"Custom model for {institute.name} saved to {model_path}")

        except Exception as e:
            db.session.rollback()
            # Re-fetch the institute object after rollback before modifying it
            institute = Institute.query.get(institute_id)
            if institute:
                institute.model_training_status = 'failed'
                db.session.commit()
            logger.error(f"Custom model training failed for institute {institute_id}: {e}", exc_info=True)

@celery.task(bind=True)
def generate_timetable_background(self, generation_id, nlp_text=None):
    """Generates a timetable using the institute's custom-trained model."""
    logger.info(f"Starting generation task {generation_id}.")

    with app.app_context():
        generation = TimetableGeneration.query.get(generation_id)
        if not generation:
            logger.error(f"Generation {generation_id} not found")
            return

        try:
            institute = Institute.query.get(generation.institute_id)
            if not (institute and institute.model_path and os.path.exists(institute.model_path)):
                raise FileNotFoundError("Institute's custom trained model is not available.")

            schema_config = json.loads(institute.config_json or '{}')

            # Try to extract real data if DB URL and config exist
            courses, faculty, rooms, map_data = None, None, None, None
            if institute.db_url and schema_config:
                try:
                    adapter = DatabaseAdapter(institute.db_url)
                    if adapter.connect():
                        courses, faculty, rooms, map_data, success = adapter.extract_data(schema_config)
                        if not success:
                            raise ValueError("Data extraction failed")
                except Exception as e:
                    logger.warning(f"Failed to extract real data for generation {generation_id}: {e}")

            # Use default data if extraction failed
            if not courses:
                courses = ['Math', 'Physics', 'Chemistry', 'Biology', 'English']
                faculty = ['Dr. Smith', 'Prof. Johnson', 'Dr. Williams', 'Prof. Brown']
                rooms = ['Room A', 'Room B', 'Room C', 'Lab 1', 'Lab 2']
                map_data = {
                    'Math': 'Dr. Smith',
                    'Physics': 'Prof. Johnson',
                    'Chemistry': 'Dr. Williams',
                    'Biology': 'Prof. Brown',
                    'English': 'Dr. Smith'
                }

            config = EnhancedConfig(DEMO_MODE=True)
            env = HybridTimetableEnv(courses, faculty, rooms, map_data, config)
            model = PPO.load(institute.model_path, env=env)

            demo = EnhancedTimetableDemo(config)
            demo.nlp_parser = NLPParser(courses, faculty, rooms)
            timetable, stats = demo.generate_hybrid_timetable(model, env, nlp_text=nlp_text)

            generation.timetable_data = json.dumps({
                'schedule': timetable.tolist() if timetable is not None else [],
                'rooms': rooms,
                'courses': {i + 1: c for i, c in enumerate(courses)},
                'faculty': faculty,
                'stats': stats
            })
            generation.task_completion_rate = stats.get('task_completion_rate', 0.0)
            generation.status = 'completed'
            db.session.commit()
            logger.info(f"Timetable generation completed for {generation_id}")

        except Exception as e:
            logger.error(f"Generation failed for {generation.id}: {e}", exc_info=True)
            db.session.rollback()
            generation = TimetableGeneration.query.get(generation_id)
            if generation:
                generation.status = 'failed'
                db.session.commit()

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def home():
    if 'user_id' in session and session.get('user_type') == 'institute':
        return redirect(url_for('dashboard'))
    elif 'user_id' in session and session.get('user_type') == 'student':
        return redirect(url_for('student_dashboard'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
            institute = Institute(
                name=request.form['institute_name'],
                email=request.form['email'],
                username=request.form['username'],
                password_hash=hashed_password,
                db_url=request.form.get('db_url')
            )
            db.session.add(institute)
            db.session.commit()
            train_custom_model.delay(institute.id)
            flash('Signup successful! Your custom AI model is now training.', 'success')
            return redirect(url_for('login'))
        except Exception:
            db.session.rollback()
            flash('Signup failed. Username or email may already be in use.', 'danger')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        institute = Institute.query.filter_by(username=request.form['username']).first()
        if institute and bcrypt.check_password_hash(institute.password_hash, request.form['password']):
            session['user_id'] = institute.id
            session['user_type'] = 'institute'
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Please check credentials.', 'danger')
    return render_template('login.html')

@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        student = Student.query.filter_by(username=request.form['username']).first()
        if student and bcrypt.check_password_hash(student.password_hash, request.form['password']):
            session['user_id'] = student.id
            session['user_type'] = 'student'
            session['institute_id'] = student.institute_id
            return redirect(url_for('student_dashboard'))
        else:
            flash('Login failed. Please check credentials.', 'danger')
    return render_template('student_login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session or session.get('user_type') != 'institute':
        return redirect(url_for('login'))
    institute = Institute.query.get(session['user_id'])
    generations = TimetableGeneration.query.filter_by(institute_id=institute.id).order_by(TimetableGeneration.generation_time.desc()).all()
    return render_template('dashboard.html', institute=institute, generations=generations)

@app.route('/student_dashboard')
def student_dashboard():
    if 'user_id' not in session or session.get('user_type') != 'student':
        return redirect(url_for('student_login'))
    student = Student.query.get(session['user_id'])
    latest_generation = TimetableGeneration.query.filter_by(institute_id=student.institute_id, status='completed').order_by(TimetableGeneration.generation_time.desc()).first()
    timetable_data = json.loads(latest_generation.timetable_data) if latest_generation else None
    return render_template('student_dashboard.html', student=student, generation=timetable_data)

@app.route('/generate-timetable', methods=['POST'])
def generate_timetable():
    if 'user_id' not in session or session.get('user_type') != 'institute': abort(401)
    institute = Institute.query.get(session['user_id'])
    if institute.model_training_status != 'completed':
        flash('Your custom model is not ready. Please wait for training to complete.', 'warning')
        return redirect(url_for('dashboard'))

    nlp_text = request.form.get('nlp_text', '')
    generation = TimetableGeneration(institute_id=institute.id, status='processing', nlp_query=nlp_text)
    db.session.add(generation)
    db.session.commit()
    generate_timetable_background.delay(generation.id, nlp_text)
    flash('New timetable generation has started.', 'info')
    return redirect(url_for('dashboard'))

# =======================
# === NEW ROUTE ADDED ===
# =======================
@app.route('/timetable-status/<int:generation_id>')
def timetable_status(generation_id):
    if 'user_id' not in session:
        abort(401)

    gen = TimetableGeneration.query.filter_by(id=generation_id, institute_id=session['user_id']).first_or_404()

    response = {'status': gen.status}
    if gen.status == 'completed' and gen.timetable_data:
        response['data'] = json.loads(gen.timetable_data)

    return jsonify(response)


@app.route('/settings')
def settings():
    if 'user_id' not in session or session.get('user_type') != 'institute': abort(401)
    institute = Institute.query.get(session['user_id'])
    students = Student.query.filter_by(institute_id=institute.id).order_by(Student.name).all()

    # ==========================
    # === SETTINGS FIX APPLIED ===
    # ==========================
    # Parse the JSON here in the backend instead of in the template
    try:
        config_data = json.loads(institute.config_json) if institute.config_json else {}
    except json.JSONDecodeError:
        config_data = {} # Handle case of invalid JSON in DB

    return render_template('settings.html', institute=institute, students=students, config=config_data)

@app.route('/settings/update-institute', methods=['POST'])
def update_institute_settings():
    if 'user_id' not in session or session.get('user_type') != 'institute': abort(401)
    institute = Institute.query.get(session['user_id'])
    institute.name = request.form['institute_name']
    institute.email = request.form['email']
    institute.username = request.form['username']
    institute.db_url = request.form['db_url']
    db.session.commit()
    flash("Institute details updated successfully.", "success")
    return redirect(url_for('settings'))

@app.route('/settings/update-schema', methods=['POST'])
def update_schema_settings():
    if 'user_id' not in session or session.get('user_type') != 'institute': abort(401)
    institute = Institute.query.get(session['user_id'])
    schema_config = {
        "courses": {"table": request.form.get('courses_table'), "col_name": request.form.get('courses_col_name')},
        "faculty": {"table": request.form.get('faculty_table'), "col_name": request.form.get('faculty_col_name')},
        "rooms": {"table": request.form.get('rooms_table'), "col_name": request.form.get('rooms_col_name')},
        "map": {"table": request.form.get('map_table'), "col_course_fk": request.form.get('map_col_course_fk'), "col_faculty_fk": request.form.get('map_col_faculty_fk')}
    }
    institute.config_json = json.dumps(schema_config)
    db.session.commit()
    # It's a good idea to trigger a re-train after schema changes
    train_custom_model.delay(institute.id)
    flash("Database schema mapping saved. AI model is now retraining with the new schema.", "info")
    return redirect(url_for('settings'))

@app.route('/settings/add-student', methods=['POST'])
def add_student():
    if 'user_id' not in session or session.get('user_type') != 'institute': abort(401)
    try:
        hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        new_student = Student(institute_id=session['user_id'], name=request.form['student_name'], username=request.form['student_username'], password_hash=hashed_password)
        db.session.add(new_student)
        db.session.commit()
        flash(f"Student '{new_student.name}' added successfully.", "success")
    except Exception:
        db.session.rollback()
        flash("A student with this username may already exist.", "danger")
    return redirect(url_for('settings'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.cli.command("init-db")
def init_database():
    """Command to initialize the database."""
    with app.app_context():
        db.create_all()
        logger.info("Database initialized successfully.")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
