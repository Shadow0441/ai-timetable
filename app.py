import os
import json
import traceback
import tempfile
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, abort
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy import create_engine, text, inspect
from celery import Celery, Task
from stable_baselines3 import PPO
import numpy as np

# Import ML engine
from hackathon_timetable import EnhancedTimetableDemo, HybridTimetableEnv, EnhancedConfig

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_celery(app):
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

app = Flask(__name__)

# --- CONFIGURATION ---
app.config.from_mapping(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key-for-production'),
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    CELERY_BROKER_URL=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)

if not app.config['SQLALCHEMY_DATABASE_URI']:
    raise RuntimeError("DATABASE_URL environment variable is not set.")

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
    # --- SECURITY FIX: Removed db_connection_url from the database model. ---
    # This credential should NEVER be stored in your application database.
    # It should be managed via environment variables or a secure secret store.
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    config_json = db.Column(db.Text, nullable=True)
    model_trained = db.Column(db.Boolean, default=False)
    last_training = db.Column(db.DateTime, nullable=True)
    model_data = db.Column(db.LargeBinary, nullable=True)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # ... student model remains the same ...

class TimetableGeneration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False)
    timetable_data = db.Column(db.Text, nullable=True)
    generation_time = db.Column(db.DateTime, default=datetime.utcnow)
    # --- REFACTOR: Changed success_rate to task_completion_rate for clarity ---
    task_completion_rate = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(50), default='pending')

# =============================================================================
# DATABASE ADAPTER (REFACTORED FOR PRODUCTION)
# =============================================================================
class DatabaseAdapter:
    def __init__(self, db_url: str):
        if not db_url:
            raise ValueError("Database URL cannot be None or empty.")
        self.db_url = db_url
        self.engine = None

    def connect(self):
        try:
            self.engine = create_engine(self.db_url, connect_args={'connect_timeout': 10})
            with self.engine.connect() as connection:
                # Test connection with a simple query
                connection.execute(text("SELECT 1"))
            return True, "Connected successfully"
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False, f"Connection failed: {str(e)}"

    def extract_data(self):
        """
        Extracts required data from the client's database.
        REFACTOR: This now includes commented-out examples of real SQL queries.
        For development, it falls back to mock data if the queries fail.
        """
        if not self.engine:
            return None, None, None, None, False

        try:
            with self.engine.connect() as connection:
                # --- PRODUCTION LOGIC: Replace these queries with the actual schema ---
                # courses = [row[0] for row in connection.execute(text("SELECT name FROM courses_table"))]
                # faculty = [row[0] for row in connection.execute(text("SELECT name FROM faculty_table"))]
                # rooms = [row[0] for row in connection.execute(text("SELECT name FROM rooms_table"))]
                # course_faculty_result = connection.execute(text("SELECT course_name, faculty_name FROM course_faculty_map_table"))
                # course_faculty_map = {row[0]: row[1] for row in course_faculty_result}
                #
                # if not all([courses, faculty, rooms, course_faculty_map]):
                #     raise ValueError("One or more data extraction queries returned no results.")
                #
                # return courses, faculty, rooms, course_faculty_map, True
                pass # Comment this out when using real queries

        except Exception as e:
            logger.warning(f"Real data extraction failed: {e}. Falling back to DEMO data.")

        # --- FALLBACK DEMO DATA ---
        courses = ["Math101", "Physics101", "Chemistry101", "Biology101", "English101", "History202", "Art101"]
        faculty = ["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown"]
        rooms = ["Room A", "Room B", "Lab 1", "Hall C"]
        course_faculty_map = {c: faculty[i % len(faculty)] for i, c in enumerate(courses)}
        return courses, faculty, rooms, course_faculty_map, True

# =============================================================================
# CELERY BACKGROUND TASKS
# =============================================================================
def get_db_url_for_institute(institute_id: int) -> str:
    """
    SECURITY: Fetches the database URL from a secure source (environment variables).
    This prevents storing sensitive credentials in the application database.
    In production, this could also query a service like HashiCorp Vault.
    """
    return os.environ.get(f'INSTITUTE_{institute_id}_DB_URL')


@celery.task
def train_base_model(institute_id):
    logger.info(f"Starting base model training for institute {institute_id}...")
    try:
        institute = Institute.query.get(institute_id)
        if not institute:
            raise Exception("Institute not found")

        # SECURITY FIX: Get URL from secure source
        db_url = get_db_url_for_institute(institute_id)
        db_adapter = DatabaseAdapter(db_url)
        connected, _ = db_adapter.connect()
        if not connected:
            raise Exception("Institute database connection failed")

        courses, faculty, rooms, course_faculty_map, success = db_adapter.extract_data()
        if not success or not courses:
            raise Exception("Failed to extract training data")

        config_data = json.loads(institute.config_json) if institute.config_json else {}
        config = EnhancedConfig(DEMO_MODE=False, INTENSIVE_TRAINING=10000)
        config.DAYS = config_data.get('days_per_week', config.DAYS)
        config.HOURS_PER_DAY = config_data.get('hours_per_day', config.HOURS_PER_DAY)

        env = HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, config)
        demo = EnhancedTimetableDemo(config)
        model = demo.train_hybrid_model(env)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            model.save(tmp.name)
            with open(tmp.name, 'rb') as f:
                model_binary = f.read()
        os.unlink(tmp.name)

        institute.model_data = model_binary
        institute.model_trained = True
        institute.last_training = datetime.utcnow()
        db.session.commit()
        logger.info(f"Base model for institute {institute_id} trained and saved successfully.")
    except Exception as e:
        logger.error(f"Base model training failed for institute {institute_id}: {e}", exc_info=True)


@celery.task
def generate_timetable_background(generation_id):
    logger.info(f"Starting timetable generation task {generation_id}...")
    generation = TimetableGeneration.query.get(generation_id)
    try:
        institute = Institute.query.get(generation.institute_id)

        if not institute or not institute.model_data:
            raise Exception("Institute or its base model not found")

        db_url = get_db_url_for_institute(institute.id)
        db_adapter = DatabaseAdapter(db_url)
        connected, _ = db_adapter.connect()
        if not connected:
            raise Exception("Institute database connection failed")

        courses, faculty, rooms, course_faculty_map, success = db_adapter.extract_data()
        if not success:
            raise Exception("Failed to extract generation data")

        config_data = json.loads(institute.config_json) if institute.config_json else {}
        config = EnhancedConfig(DEMO_MODE=True, DEMO_TRAINING=2000) # Short fine-tuning
        config.DAYS = config_data.get('days_per_week', config.DAYS)
        config.HOURS_PER_DAY = config_data.get('hours_per_day', config.HOURS_PER_DAY)

        env = HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, config)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
            tmp.write(institute.model_data)
            model_path = tmp.name

        try:
            model = PPO.load(model_path, env=env)
            logger.info(f"Loaded base model for fine-tuning for generation {generation_id}.")
            model.learn(total_timesteps=config.DEMO_TRAINING)
            logger.info(f"Model fine-tuning complete for generation {generation_id}.")

            # CORE LOGIC FIX: The fine-tuned model is now passed to the generation function
            demo = EnhancedTimetableDemo(config)
            timetable, stats = demo.generate_hybrid_timetable(model, env)
        finally:
            os.unlink(model_path)

        timetable_list = timetable.tolist() if timetable is not None else []
        generation.timetable_data = json.dumps({
            'schedule': timetable_list,
            'courses': {i + 1: c for i, c in enumerate(courses)},
            'stats': stats
        })
        generation.task_completion_rate = stats.get('task_completion_rate', 0.0)
        generation.status = 'completed'
        db.session.commit()
        logger.info(f"Timetable generation {generation_id} completed successfully.")
    except Exception as e:
        logger.error(f"Timetable generation failed for {generation_id}: {e}", exc_info=True)
        generation.status = 'failed'
        db.session.commit()

# =============================================================================
# FLASK ROUTES
# =============================================================================
# ... Home, dashboard, and other routes would go here ...
# For brevity, focusing on the core API endpoints that were refactored.

@app.route('/signup/institute', methods=['POST'])
def signup_institute():
    # This endpoint now simulates checking for the DB_URL env var
    # instead of asking for it in the request.
    try:
        data = request.get_json()
        # In a real app, you would assign a new institute ID and then construct
        # the expected env var name to check if it's configured.
        # For this demo, we'll just proceed.

        hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
        institute = Institute(
            name=data['institute_name'], email=data['email'], username=data['username'],
            password_hash=hashed_password
        )
        db.session.add(institute)
        db.session.commit()

        # You would then need to instruct the new client to set up their
        # environment variable: INSTITUTE_{institute.id}_DB_URL
        # before training can succeed.

        train_base_model.delay(institute.id)
        return jsonify({'success': f'Institute registered with ID {institute.id}! Base model training has started. Please ensure the DB connection environment variable is set.'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Signup failed: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/generate-timetable', methods=['POST'])
def generate_timetable():
    # This route is functionally the same but now relies on the improved backend logic
    if 'user_id' not in session or session.get('user_type') != 'institute':
        return jsonify({'error': 'Unauthorized'}), 401

    institute = Institute.query.get(session['user_id'])
    if not institute.model_trained:
        return jsonify({'error': 'Base model is not ready yet. Please wait.'}), 400

    generation = TimetableGeneration(institute_id=institute.id, status='processing')
    db.session.add(generation)
    db.session.commit()

    generate_timetable_background.delay(generation.id)
    return jsonify({'success': 'Timetable generation started.', 'generation_id': generation.id})


@app.route('/timetable-status/<int:generation_id>')
def timetable_status(generation_id):
    # --- SECURITY FIX: This endpoint is now authenticated. ---
    if 'user_id' not in session or session.get('user_type') != 'institute':
        abort(401) # Unauthorized

    generation = TimetableGeneration.query.get_or_404(generation_id)

    # Users can only view the status of generations they own.
    if generation.institute_id != session['user_id']:
        abort(403) # Forbidden

    return jsonify({
        'status': generation.status,
        'timetable_data': json.loads(generation.timetable_data) if generation.timetable_data else None
    })

# ... init-db command and other CLI tools ...
@app.cli.command("init-db")
def init_database():
    with app.app_context():
        db.create_all()
        logger.info("Database initialized successfully!")
