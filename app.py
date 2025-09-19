import os
import json
import io
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv

from flask import (Flask, render_template, request, session,
                   redirect, url_for, abort, flash, jsonify, make_response)
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy import exc

from connection import connect_and_fetch, filter_data_for_parser
from nlp_parser import NLP_Parser
from solver import ConstraintSolver
from config import Config

load_dotenv()

app = Flask(__name__)

# --- Configuration ---
app.config.from_mapping(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'a-very-secret-key'),
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///site.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)

# --- Initializations ---
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# =============================================================================
# DATABASE MODELS
# =============================================================================
class Institute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(60), nullable=False)
    db_url = db.Column(db.String(255), nullable=True)
    config_json = db.Column(db.Text, nullable=True)

    students = db.relationship('Student', backref='institute', lazy=True, cascade="all, delete-orphan")
    settings = db.relationship('InstituteSetting', backref='institute', lazy=True, uselist=False, cascade="all, delete-orphan")
    timetables = db.relationship('Timetable', backref='institute', lazy=True, cascade="all, delete-orphan")

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    admission_id = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(60), nullable=False)
    course = db.Column(db.String(100), nullable=False)
    branch = db.Column(db.String(100), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False)

class InstituteSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False, unique=True)
    total_rooms = db.Column(db.Integer, default=10)
    total_labs = db.Column(db.Integer, default=5)
    start_time = db.Column(db.String(5), default="09:00")
    end_time = db.Column(db.String(5), default="17:00")
    lunch_duration_hr = db.Column(db.Integer, default=1)
    lecture_duration_hr = db.Column(db.Integer, default=1)

class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    institute_id = db.Column(db.Integer, db.ForeignKey('institute.id'), nullable=False)
    course = db.Column(db.String(100), nullable=False)
    branch = db.Column(db.String(100), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    timetable_json = db.Column(db.Text, nullable=False)
    is_published = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route("/")
@app.route("/home")
def home():
    if 'institute_id' in session:
        return redirect(url_for('dashboard'))
    if 'student_id' in session:
        return redirect(url_for('student_dashboard'))
    return render_template('home_page.html')

@app.route("/institute/signup", methods=['GET', 'POST'])
def institute_signup():
    if request.method == 'POST':
        hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        institute = Institute(
            name=request.form['name'],
            username=request.form['username'],
            email=request.form['email'],
            password_hash=hashed_password,
            db_url=request.form['db_url']
        )
        db.session.add(institute)
        db.session.commit()

        settings = InstituteSetting(institute_id=institute.id)
        db.session.add(settings)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('institute_login'))
    return render_template('institute_signup.html')

@app.route("/institute/login", methods=['GET', 'POST'])
def institute_login():
    if request.method == 'POST':
        institute = Institute.query.filter_by(username=request.form['username']).first()
        if institute and bcrypt.check_password_hash(institute.password_hash, request.form['password']):
            session['institute_id'] = institute.id
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('intitute_login.html')

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route("/dashboard")
def dashboard():
    if 'institute_id' not in session:
        return redirect(url_for('institute_login'))

    institute = Institute.query.get(session['institute_id'])

    if not institute:
        session.clear()
        flash('Your session was invalid. Please log in again.', 'warning')
        return redirect(url_for('institute_login'))

    timetables = Timetable.query.filter_by(institute_id=institute.id).order_by(Timetable.created_at.desc()).all()
    return render_template('dashboard.html', institute=institute, timetables=timetables)


@app.route("/generate", methods=['POST'])
def generate_timetable():
    if 'institute_id' not in session:
        abort(403)

    institute = Institute.query.get(session['institute_id'])
    if not institute:
        return redirect(url_for('logout'))

    settings = institute.settings
    course = request.form['course']
    branch_input = request.form['branch']
    year = int(request.form['year'])
    nlp_text = request.form['nlp_text']

    try:
        schema_config = json.loads(institute.config_json) if institute.config_json else None
        full_df, success, message = connect_and_fetch(institute.db_url, schema_config)

        if not success:
            raise ConnectionError(message)
        flash(message, 'info')

        parser_data, matched_branch = filter_data_for_parser(full_df, branch_value=branch_input, year_value=year)

        if not parser_data:
            raise ValueError(f"No data found for branch '{branch_input}' and year '{year}'. Check inputs and schema configuration.")

        nlp_parser = NLP_Parser(parser_data)
        nlp_constraints = nlp_parser.parse(nlp_text)

        config = Config()
        solver = ConstraintSolver(config, settings)

        faculty_subject_map = {subj: fac for fac, subjs in parser_data['faculty_subjects'].items() for subj in subjs}

        subjects_for_solver = parser_data['branch_subjects'][(matched_branch, year)]

        timetable_array, stats = solver.solve(
            subjects=subjects_for_solver,
            faculty=list(parser_data['faculty_subjects'].keys()),
            rooms=parser_data['rooms'],
            faculty_map=faculty_subject_map,
            num_lectures_map=parser_data['num_lectures'],
            nlp_constraints=nlp_constraints
        )

        if timetable_array is None:
            raise Exception(f"Solver failed: {stats.get('reason', 'Could not find a feasible solution.')}")

        time_slots = []
        start_time = datetime.strptime(settings.start_time, '%H:%M')
        for i in range(solver.hours_per_day):
            current_time = start_time + timedelta(hours=i)
            next_time = start_time + timedelta(hours=i + 1)
            time_slots.append(f"{current_time.strftime('%I:%M %p')} - {next_time.strftime('%I:%M %p')}")

        subject_index_map = {str(i + 1): name for i, name in enumerate(subjects_for_solver)}

        timetable_to_save = {
            "schedule": timetable_array.tolist(),
            "subjects": subject_index_map,
            "rooms": parser_data['rooms'],
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][:config.DAYS],
            "time_slots": time_slots
        }

        new_timetable = Timetable(
            institute_id=institute.id,
            course=course,
            branch=matched_branch,
            year=year,
            timetable_json=json.dumps(timetable_to_save)
        )
        db.session.add(new_timetable)
        db.session.commit()
        flash('Timetable generated successfully!', 'success')

    except Exception as e:
        flash(f'An error occurred: {e}', 'danger')

    return redirect(url_for('dashboard'))

@app.route("/timetable/<int:timetable_id>/view")
def view_timetable(timetable_id):
    if 'institute_id' not in session:
        return jsonify({"error": "Unauthorized"}), 403

    timetable = Timetable.query.get_or_404(timetable_id)

    if timetable.institute_id != session['institute_id']:
        return jsonify({"error": "Access Denied"}), 403

    return jsonify(json.loads(timetable.timetable_json))

@app.route("/timetable/<int:timetable_id>/export")
def export_timetable(timetable_id):
    if 'institute_id' not in session:
        abort(403)

    timetable = Timetable.query.get_or_404(timetable_id)

    if timetable.institute_id != session['institute_id']:
        abort(403)

    data = json.loads(timetable.timetable_json)

    output = io.StringIO()
    writer = csv.writer(output)

    header = ['Time'] + data['days']
    writer.writerow(header)

    for h, time_slot in enumerate(data['time_slots']):
        row = [time_slot]
        for d in range(len(data['days'])):
            cell_content = []
            schedule_day = data['schedule'][d]
            if schedule_day and len(schedule_day) > h and schedule_day[h]:
                for r, subject_index in enumerate(schedule_day[h]):
                    if subject_index > 0:
                        subject_name = data['subjects'].get(str(subject_index), 'N/A')
                        room_name = data['rooms'][r]
                        cell_content.append(f"{subject_name} ({room_name})")
            row.append(" | ".join(cell_content))
        writer.writerow(row)

    output.seek(0)

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = f"attachment; filename=timetable_{timetable.branch.replace(' ','_')}_{timetable.year}.csv"
    response.headers["Content-type"] = "text/csv"

    return response


@app.route("/timetable/<int:timetable_id>/publish")
def publish_timetable(timetable_id):
    timetable = Timetable.query.get_or_404(timetable_id)
    if timetable.institute_id != session.get('institute_id'):
        abort(403)
    timetable.is_published = True
    db.session.commit()
    flash(f'Timetable for {timetable.branch} (Year {timetable.year}) has been published.', 'success')
    return redirect(url_for('dashboard'))

@app.route("/settings", methods=['GET', 'POST'])
def settings():
    if 'institute_id' not in session:
        return redirect(url_for('institute_login'))

    institute = Institute.query.get(session['institute_id'])

    if not institute:
        session.clear()
        return redirect(url_for('institute_login'))

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'general_settings':
            settings = institute.settings
            settings.total_rooms = request.form['total_rooms']
            settings.total_labs = request.form['total_labs']
            settings.start_time = request.form['start_time']
            settings.end_time = request.form['end_time']
            flash('General settings updated!', 'success')

        elif form_type == 'schema_settings':
            schema_config = {
                'subjects': {'table': request.form['s_table'], 'name': request.form['s_name'], 'branch_fk': request.form['s_branch_fk'], 'year': request.form['s_year']},
                'branches': {'table': request.form['b_table'], 'name': request.form['b_name'], 'course_fk': request.form['b_course_fk']},
                'courses': {'table': request.form['c_table'], 'name': request.form['c_name']},
                'faculty': {'table': request.form['f_table'], 'name': request.form['f_name']},
                'rooms': {'table': request.form['r_table'], 'name': request.form['r_name']},
                'map': {'table': request.form['map_table'], 'subject_fk': request.form['map_s_fk'], 'faculty_fk': request.form['map_f_fk']}
            }
            institute.config_json = json.dumps(schema_config)
            flash('Database schema mapping saved!', 'success')

        db.session.commit()
        return redirect(url_for('settings'))

    current_config = json.loads(institute.config_json) if institute.config_json else {}
    return render_template('settings.html', institute=institute, config=current_config)

@app.route("/settings/add_student", methods=['POST'])
def add_student():
    if 'institute_id' not in session:
        abort(403)

    institute = Institute.query.get(session['institute_id'])
    if not institute:
        return redirect(url_for('logout'))

    hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
    student = Student(
        name=request.form['name'],
        admission_id=request.form['admission_id'],
        password_hash=hashed_password,
        course=request.form['course'],
        branch=request.form['branch'],
        year=int(request.form['year']),
        institute_id=session['institute_id']
    )
    db.session.add(student)
    db.session.commit()
    flash('Student added successfully!', 'success')
    return redirect(url_for('settings'))

@app.route("/student/login", methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        student = Student.query.filter_by(admission_id=request.form['admission_id']).first()
        if student and bcrypt.check_password_hash(student.password_hash, request.form['password']):
            session['student_id'] = student.id
            return redirect(url_for('student_dashboard'))
        else:
            flash('Login Unsuccessful. Please check Admission ID and password', 'danger')
    return render_template('student_login.html')

@app.route("/student/dashboard")
def student_dashboard():
    if 'student_id' not in session:
        return redirect(url_for('student_login'))

    student = Student.query.get(session['student_id'])

    if not student:
        session.clear()
        return redirect(url_for('student_login'))

    timetable = Timetable.query.filter_by(
        institute_id=student.institute_id,
        course=student.course,
        branch=student.branch,
        year=student.year,
        is_published=True
    ).order_by(Timetable.created_at.desc()).first()

    timetable_data = json.loads(timetable.timetable_json) if timetable else None

    return render_template('student_dashboard.html', student=student, timetable_data=timetable_data)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


@app.cli.command("init-db")
def init_db_command():
    """Creates the database tables and a sample institute for testing."""
    db.drop_all()
    db.create_all()

    institute_pass = bcrypt.generate_password_hash('password').decode('utf-8')
    sample_institute = Institute(
        name='Sample Institute',
        email='test@institute.com',
        username='tester',
        password_hash=institute_pass,
        db_url='sqlite:///sample_institute.db'
    )
    db.session.add(sample_institute)
    db.session.commit()

    settings = InstituteSetting(institute_id=sample_institute.id)
    db.session.add(settings)
    db.session.commit()

    print("Initialized the main database (site.db).")
    print("Added a 'Sample Institute' for testing.")
    print("Login with: username='tester', password='password'")

