import pandas as pd
from sqlalchemy import create_engine, text, exc, inspect
import argparse
import json

def build_query_from_map(schema_config):
    """Dynamically builds a SQL JOIN query from a user-defined schema map."""
    s, b, c, f, r, m = [schema_config.get(k, {}) for k in ['subjects', 'branches', 'courses', 'faculty', 'rooms', 'map']]

    if not all([s.get('table'), s.get('name'), b.get('table'), b.get('name'), f.get('table'), f.get('name')]):
        return None, "Manual schema config is incomplete. Please define at least subjects, branches, and faculty tables and name columns."

    query = f"""
    SELECT
        s.{s['name']} AS subject,
        b.{b['name']} AS branch,
        s.{s.get('year', 'year_id')} AS year_of_study,
        c.{c.get('name', 'course_name')} AS course,
        f.{f.get('name', 'full_name')} AS faculty,
        r.{r.get('name', 'room_name')} AS room,
        s.{s.get('lectures', 'num_lectures')} AS num_lectures
    FROM
        {s['table']} s
    LEFT JOIN {b['table']} b ON s.{s.get('branch_fk', 'branch_id')} = b.{b.get('id', 'branch_id')}
    LEFT JOIN {c.get('table', 'courses')} c ON b.{b.get('course_fk', 'course_id')} = c.{c.get('id', 'course_id')}
    LEFT JOIN {m.get('table', 'enrollments')} m ON s.{s.get('id', 'subject_id')} = m.{m.get('subject_fk', 'subject_id')}
    LEFT JOIN {f['table']} f ON m.{m.get('faculty_fk', 'teacher_id')} = f.{f.get('id', 'teacher_id')}
    LEFT JOIN {r.get('table', 'rooms')} r ON s.{s.get('room_fk', 'preferred_room_id')} = r.{r.get('id', 'room_id')}
    """
    return query, "Successfully built query from manual schema."


def get_auto_detect_query(engine, schema_prefix):
    """Builds the query for the known 'public' or 'university' schema."""
    return f"""
    SELECT
        s.subject_name AS subject,
        b.branch_name AS branch,
        ay.year_of_study AS year_of_study,
        c.course_name AS course,
        t.full_name AS faculty,
        r.room_name AS room,
        s.num_lectures AS num_lectures
    FROM
        {schema_prefix}.subjects s
    LEFT JOIN {schema_prefix}.branches b ON s.branch_id = b.branch_id
    LEFT JOIN {schema_prefix}.courses c ON b.course_id = c.course_id
    LEFT JOIN {schema_prefix}.academic_years ay ON s.year_id = ay.year_id
    LEFT JOIN {schema_prefix}.enrollments e ON s.subject_id = e.subject_id
    LEFT JOIN {schema_prefix}.teachers t ON e.teacher_id = t.teacher_id
    LEFT JOIN {schema_prefix}.rooms r ON 1=1 -- Simple join to get all rooms, can be refined
    """

def connect_and_fetch(db_uri, schema_config=None):
    """
    Connects to a database, attempting to auto-detect a known schema first,
    then falling back to a user-provided schema map.
    """
    engine = create_engine(db_uri)
    query = None
    status_message = ""

    try:
        inspector = inspect(engine)
        if inspector.has_table("subjects", schema="public"):
            query = get_auto_detect_query(engine, "public")
            status_message = "Successfully auto-detected 'public' schema."
        elif inspector.has_table("subjects", schema="university"):
            query = get_auto_detect_query(engine, "university")
            status_message = "Successfully auto-detected 'university' schema."
    except Exception as e:
        print(f"Schema auto-detection failed: {e}")

    if not query and schema_config:
        query, status_message = build_query_from_map(schema_config)
        if not query:
            return None, False, status_message

    if not query:
        return None, False, "Could not determine which query to use. Please configure schema in settings."

    try:
        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)
        if df.empty:
            return df, True, "Query executed successfully but returned no data. Check database content and schema."
        return df, True, status_message
    except exc.SQLAlchemyError as e:
        error_message = f"Database Error: {e.orig}. Please check your connection URL and schema configuration."
        return None, False, error_message
    except Exception as e:
        return None, False, f"An unexpected error occurred: {e}"


def filter_data_for_parser(df, branch_value, year_value):
    """
    Cleans and structures a DataFrame, returning the structured data
    and the actual matched branch name.
    """
    if df is None or df.empty:
        return None, None

    user_branch_lower = branch_value.lower()
    unique_branches = df['branch'].dropna().unique()
    matched_branch = None

    for item in unique_branches:
        if item.lower() == user_branch_lower:
            matched_branch = item
            break
    if not matched_branch:
        possible_matches = [item for item in unique_branches if user_branch_lower in item.lower()]
        if len(possible_matches) == 1: matched_branch = possible_matches[0]
    if not matched_branch:
        for item in unique_branches:
            abbreviation = "".join(word[0] for word in item.replace('-', ' ').split()).lower()
            if abbreviation == user_branch_lower:
                matched_branch = item
                break

    if not matched_branch:
        return None, None

    filtered_df = df[(df['branch'] == matched_branch) & (df['year_of_study'] == year_value)].copy()

    if filtered_df.empty:
        return None, matched_branch

    branch_subjects = {
        (branch, year): group['subject'].unique().tolist()
        for (branch, year), group in filtered_df.groupby(['branch', 'year_of_study'])
    }

    faculty_subjects = {
        faculty: group['subject'].unique().tolist()
        for faculty, group in filtered_df.groupby('faculty') if pd.notna(faculty)
    }

    filtered_data = {
        "courses": filtered_df['course'].dropna().unique().tolist(),
        "branch": filtered_df['branch'].dropna().unique().tolist(),
        "branch_subjects": branch_subjects,
        "faculty_subjects": faculty_subjects,
        "rooms": df['room'].dropna().unique().tolist(),
        "num_lectures": filtered_df.set_index('subject')['num_lectures'].to_dict()
    }

    return filtered_data, matched_branch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch and filter timetable data from a database.")
    parser.add_argument("db_uri", help="SQLAlchemy database URI (e.g., 'sqlite:///timetable.db')")
    parser.add_argument("--branch", help="Branch to filter for (e.g., 'Computer Science').")
    parser.add_argument("--year", type=int, help="Year to filter for (e.g., 2).")
    args = parser.parse_args()

    full_df, success, message = connect_and_fetch(args.db_uri)

    if success:
        print(f"\n--- Connection Status ---\n{message}")
        if args.branch and args.year:
            filtered_data, matched_name = filter_data_for_parser(full_df, branch_value=args.branch, year_value=args.year)
            print(f"\nMatched Branch Name: {matched_name}")
            print("\n--- Filtered Data for Parser ---")
            print(json.dumps(filtered_data, indent=2))
        else:
            print("\n--- Full Data Fetched from DB ---")
            print(full_df.head())
    else:
        print(f"\n--- Operation Failed ---\n{message}")

