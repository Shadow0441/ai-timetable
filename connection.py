
import pandas as pd
from sqlalchemy import create_engine, text, exc, inspect
import argparse
import json
from typing import Optional, Tuple, Dict, Any, List

def build_query_from_map(schema_config: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Dynamically builds a SQL JOIN query from a user-defined schema map.

    Args:
        schema_config: A dictionary defining table and column names.

    Returns:
        A tuple containing the SQL query string (or None on failure) and a status message.
    """
    # This check prevents the "'str' object has no attribute 'get'" error.
    if not isinstance(schema_config, dict):
        return None, "Database schema configuration is invalid; it should be a dictionary-like object."

    s, b, c, f, r, m = [schema_config.get(k, {}) for k in ['subjects', 'branches', 'courses', 'faculty', 'rooms', 'map']]

    if not all([s.get('table'), s.get('name'), b.get('table'), b.get('name'), f.get('table'), f.get('name')]):
        return None, "Schema config incomplete. Define at least subjects, branches, and faculty tables/columns."

    # Using .get() with fallbacks makes the user's config more flexible
    query = f"""
    SELECT
        s.{s['name']} AS subject,
        b.{b['name']} AS branch,
        s.{s.get('year', 'year_of_study')} AS year_of_study,
        c.{c.get('name', 'course_name')} AS course,
        f.{f['name']} AS faculty,
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


def get_auto_detect_query(schema_prefix: str) -> str:
    """Builds the query for a known, standard schema (e.g., 'public')."""
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
    LEFT JOIN {schema_prefix}.rooms r ON 1=1
    """

def connect_and_fetch(db_uri: str, schema_config: Optional[Dict[str, Any]] = None) -> Tuple[Optional[pd.DataFrame], bool, str]:
    """
    Connects to a database, fetches academic data, and returns it as a DataFrame.

    It first attempts to auto-detect a known schema, then falls back to the user-provided map.

    Args:
        db_uri: The SQLAlchemy database connection string.
        schema_config: A dictionary defining a custom database schema.

    Returns:
        A tuple containing (DataFrame or None, success_boolean, status_message).
    """
    try:
        engine = create_engine(db_uri)
        inspector = inspect(engine)
    except Exception as e:
        return None, False, f"Invalid Database URL: {e}"

    query: Optional[str] = None
    status_message = ""

    # 1. Attempt to auto-detect a known schema
    try:
        if inspector.has_table("subjects", schema="public"):
            query = get_auto_detect_query("public")
            status_message = "Successfully auto-detected 'public' schema."
        elif inspector.has_table("subjects", schema="university"):
            query = get_auto_detect_query("university")
            status_message = "Successfully auto-detected 'university' schema."
    except Exception as e:
        print(f"Schema auto-detection failed: {e}") # Log for debugging

    # 2. If auto-detect fails, fall back to the user-provided manual schema
    if not query and schema_config:
        query, status_message = build_query_from_map(schema_config)
        if not query:
            return None, False, status_message

    if not query:
        return None, False, "Could not determine schema. Please configure it in settings."

    # 3. Execute the chosen query
    try:
        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)
        if df.empty:
            return df, True, "Query executed but returned no data. Check database content."
        return df, True, status_message
    except exc.SQLAlchemyError as e:
        error_message = f"Database Error: {e.orig}. Check your connection URL and schema config."
        return None, False, error_message
    except Exception as e:
        return None, False, f"An unexpected error occurred: {e}"


def filter_data_for_parser(df: pd.DataFrame, branch_value: str, year_value: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Filters the main DataFrame for a specific branch/year and structures it for the solver.

    Args:
        df: The complete DataFrame from the database.
        branch_value: The user-provided branch name (e.g., "Computer Science").
        year_value: The user-provided year (e.g., 2).

    Returns:
        A tuple containing (structured_data_dict or None, matched_branch_name or None).
    """
    if df is None or df.empty:
        return None, None

    # This multi-pass logic makes branch matching flexible (full name, partial, or abbreviation)
    user_branch_lower = branch_value.lower()
    unique_branches: List[str] = df['branch'].dropna().unique().tolist()
    matched_branch: Optional[str] = None

    for b in unique_branches:
        if b.lower() == user_branch_lower:
            matched_branch = b
            break
    if not matched_branch:
        possible_matches = [b for b in unique_branches if user_branch_lower in b.lower()]
        if len(possible_matches) == 1: matched_branch = possible_matches[0]
    if not matched_branch:
        for b in unique_branches:
            abbreviation = "".join(word[0] for word in b.replace('-', ' ').split()).lower()
            if abbreviation == user_branch_lower:
                matched_branch = b
                break

    if not matched_branch:
        return None, None

    filtered_df = df[(df['branch'] == matched_branch) & (df['year_of_study'] == year_value)].copy()

    if filtered_df.empty:
        return None, matched_branch

    # Structure data into the format required by the solver
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
        "rooms": df['room'].dropna().unique().tolist(), # Get all rooms from institute
        "num_lectures": filtered_df.set_index('subject')['num_lectures'].to_dict()
    }

    return filtered_data, matched_branch


# This block allows the script to be run from the command line for testing.
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
