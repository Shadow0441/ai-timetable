import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
import re

# Load the small English model. It's efficient and has all we need for this task.
nlp = spacy.load("en_core_web_sm")

class NLP_Parser():
    """
    Parses natural language text to extract structured constraints for a
    timetabling problem, suitable for use with Google OR-Tools.
    """
    def __init__(self, data):
        """
        Initializes the parser with the known data about the university,
        such as courses, branches, subjects, faculty, and rooms.
        """
        print("Initializing NLP Parser...")
        # --- Store Raw Data ---
        self.courses = data.get("courses", [])
        self.branch = data.get("branch", [])
        self.branch_subjects = data.get("branch_subjects", {})
        self.faculty_subjects = data.get("faculty_subjects", {})
        self.rooms = data.get("rooms", [])
        self.nol = data.get("num_lectures", {})

        # --- Pre-process data for efficient matching ---
        self.all_subjects = self._get_all_unique_values(self.branch_subjects)
        self.faculty_names = list(self.faculty_subjects.keys())
        self.days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

        # --- Initialize spaCy PhraseMatcher for multi-word entities ---
        self.phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        entity_map = {
            "SUBJECT": self.all_subjects,
            "BRANCH": self.branch,
            "FACULTY": self.faculty_names,
            "ROOM": self.rooms,
            "DAY": self.days_of_week
        }
        for label, terms in entity_map.items():
            patterns = [nlp.make_doc(term) for term in terms]
            self.phrase_matcher.add(label, patterns)

        # --- Initialize spaCy Matcher for rule-based patterns ---
        self.matcher = Matcher(nlp.vocab)
        self._create_patterns()
        print("Parser initialized successfully.")

    def _get_all_unique_values(self, data_dict):
        """Helper to flatten a dictionary of lists into a single set of unique values."""
        all_values = set()
        for key in data_dict:
            for value in data_dict[key]:
                all_values.add(value)
        return list(all_values)

    def _create_patterns(self):
        """
        Defines the grammatical patterns that work on pre-identified ENTITIES.
        """
        # PATTERN 1: Assign a specific room to a subject or branch.
        pattern_room = [
            {"ENT_TYPE": {"IN": ["SUBJECT", "BRANCH"]}},
            {"OP": "*"},
            {"LOWER": {"IN": ["in", "in room", "at"]}},
            {"ENT_TYPE": "ROOM"}
        ]
        self.matcher.add("ROOM_CONSTRAINT", [pattern_room])

        # PATTERN 2: Assign a faculty to a subject.
        pattern_faculty_subject = [
            {"ENT_TYPE": "FACULTY"},
            {"LEMMA": "teach"},
            {"ENT_TYPE": "SUBJECT"}
        ]
        self.matcher.add("FACULTY_SUBJECT_CONSTRAINT", [pattern_faculty_subject])

        # PATTERN 3: Time constraint for a subject/branch on a specific day.
        # Handles "on Tuesday", "on Tuesday and Thursday", "not on Friday" etc.
        pattern_day = [
             {"ENT_TYPE": {"IN": ["SUBJECT", "BRANCH"]}},
             {"OP": "*"},
             {"LOWER": {"IN": ["on", "be"]}},
             {"OP": "?"}, # handles "must not be on"
             {"ENT_TYPE": "DAY"}
        ]
        self.matcher.add("DAY_CONSTRAINT", [pattern_day])

        # PATTERN 4: Negative time constraint (still token-based).
        pattern_time_negation = [
            {"LOWER": "no"},
            {"LEMMA": {"IN": ["class", "lecture"]}},
            {"LOWER": {"IN": ["after", "before"]}},
            {"LIKE_NUM": True}
        ]
        self.matcher.add("TIME_NEGATION_CONSTRAINT", [pattern_time_negation])

        # PATTERN 5: Lecture duration/consecutive lectures.
        pattern_duration = [
            {"ENT_TYPE": "SUBJECT"},
            {"OP": "*"},
            {"LIKE_NUM": True},
            {"LEMMA": {"IN": ["hour", "lecture"]}}
        ]
        self.matcher.add("DURATION_CONSTRAINT", [pattern_duration])

    def parse(self, text):
        """
        Main method to parse a text string. It runs the phrase matcher,
        then the rule-based matcher, and dispatches to handler functions.
        """
        doc = nlp(text)

        # Step 1: Run PhraseMatcher to find all potential entities.
        phrase_matches = self.phrase_matcher(doc)

        # Step 2: Create spans from phrase matches, filtering overlaps to keep the longest match.
        sorted_matches = sorted(phrase_matches, key=lambda m: m[2] - m[1], reverse=True)
        filtered_spans = []
        seen_tokens = set()
        for match_id, start, end in sorted_matches:
            if start not in seen_tokens and end - 1 not in seen_tokens:
                label = nlp.vocab.strings[match_id]
                span = Span(doc, start, end, label=label)
                filtered_spans.append(span)
                seen_tokens.update(range(start, end))
        doc.ents = filtered_spans

        # Step 3: Run the rule-based Matcher on the doc with our custom entities.
        matches = self.matcher(doc)

        constraints = []
        matched_spans = set()
        for match_id, start, end in matches:
            span_hash = (start, end)
            if span_hash in matched_spans:
                continue
            matched_spans.add(span_hash)

            rule_name = nlp.vocab.strings[match_id]
            span = doc[start:end]

            # Dispatch to the correct processing function based on the matched rule
            handler = getattr(self, f"_process_{rule_name.lower()}", None)
            if handler:
                constraint = handler(span)
                if constraint:
                    constraints.append(constraint)

        return constraints

    # --- Handler Functions to Process Matches ---

    def _process_room_constraint(self, span):
        target, room, constraint_on = None, None, None
        for ent in span.ents:
            if ent.label_ in ["SUBJECT", "BRANCH"]:
                target = ent.text
                constraint_on = "branch_room" if ent.label_ == "BRANCH" else "subject_room"
            if ent.label_ == "ROOM":
                room = ent.text
        if target and room:
            return {"type": "room_assignment", "target": target, "room": room, "constraint_on": constraint_on}
        return {}

    def _process_faculty_subject_constraint(self, span):
        faculty, subject = None, None
        for ent in span.ents:
            if ent.label_ == "FACULTY":
                faculty = ent.text
            if ent.label_ == "SUBJECT":
                subject = ent.text
        if faculty and subject:
            return {"type": "faculty_assignment", "faculty": faculty, "subject": subject}
        return {}

    def _process_day_constraint(self, span):
        target, constraint_on = None, None
        days = [ent.text.lower() for ent in span.ents if ent.label_ == "DAY"]
        negation = "not" in span.text.lower()

        for ent in span.ents:
            if ent.label_ in ["SUBJECT", "BRANCH"]:
                target = ent.text
                constraint_on = "branch_day" if ent.label_ == "BRANCH" else "subject_day"

        if target and days:
            return {"type": "day_restriction", "target": target, "days": days,
                    "constraint_on": constraint_on, "restriction": "disallow" if negation else "allow"}
        return {}

    def _process_time_negation_constraint(self, span):
        condition, time = None, None
        for token in span:
            if token.lower_ in ["after", "before"]:
                condition = token.lower_
            if token.like_num:
                time_suffix = span[-1].lower_ if span[-1].lower_ in ["am", "pm"] else ""
                time = f"{token.text} {time_suffix}".strip()
        if condition and time:
            return {"type": "time_restriction", "restriction": "disallow", "condition": condition, "time": time}
        return {}

    def _process_duration_constraint(self, span):
        subject, duration = None, None
        for ent in span.ents:
            if ent.label_ == "SUBJECT":
                subject = ent.text
        for token in span:
            if token.like_num:
                duration = int(token.text)
        if subject and duration:
            return {"type": "lecture_duration", "subject": subject, "duration_hours": duration}
        return {}


# --- MAIN EXECUTION BLOCK (Example Usage) ---
if __name__ == '__main__':
    master_data = {
        "courses": ["B.Tech", "M.Tech"],
        "branch": ["Computer Science", "Mechanical", "ECE"],
        "branch_subjects": {
            "Computer Science": ["Data Structures", "AI", "Machine Learning"],
            "Mechanical": ["Thermodynamics", "Fluid Mechanics"],
            "ECE": ["Signals and Systems", "Digital Circuits"]
        },
        "faculty_subjects": {
            "Professor Sharma": ["Data Structures", "AI"],
            "Dr. Gupta": ["Thermodynamics", "Fluid Mechanics"],
            "Ms. Verma": ["Signals and Systems", "Digital Circuits", "Machine Learning"]
        },
        "rooms": ["C-101", "C-102", "M-201", "Auditorium", "Lab-A"],
        "num_lectures": { "Data Structures": 4, "AI": 3, "Machine Learning": 3, "Thermodynamics": 5 }
    }

    parser = NLP_Parser(master_data)

    constraints_text = [
        "Schedule all Computer Science classes in room C-101.",
        "Thermodynamics lectures should be held in the Auditorium.",
        "Professor Sharma teaches AI.",
        "No classes for any branch after 5 pm.",
        "ECE should have classes on Tuesday and Thursday only.",
        "The Machine Learning lab must be 3 hours long.",
        "Fluid Mechanics must not be on Friday.",
        "An unknown constraint that will be ignored."
    ]

    print("\n--- Parsing Constraints ---")
    all_parsed_constraints = []
    for text in constraints_text:
        parsed = parser.parse(text)
        print(f"Input: '{text}'")
        if parsed:
            print(f"  Output -> {parsed}")
            all_parsed_constraints.extend(parsed)
        else:
            print("  Output -> No constraint pattern matched.")
        print("-" * 20)

    print("\n--- Final Structured Constraints for Solver ---")
    import json
    print(json.dumps(all_parsed_constraints, indent=2))

