import re
import spacy
from rapidfuzz import process, fuzz

class NLP_Parser:
    def __init__(self, parser_data = None):
        self.nlp = spacy.load("en_core_web_sm")
        # Store context if it's provided
        if parser_data:
            self.subjects = [s for subjects_list in parser_data.get('branch_subjects', {}).values() for s in subjects_list]
            self.faculty = list(parser_data.get('faculty_subjects', {}).keys())
        else:
            self.subjects = []
            self.faculty = []

        # Constraint patterns with synonyms and regex
        self.constraint_patterns = {
            "no_classes_after": [
                r"(no|avoid|dont|don’t).*(class|lecture|lectures).*after (\d{1,2})(?: ?(am|pm))?",
            ],
            "no_classes_before": [
                r"(no|avoid|dont|don’t).*(class|lecture|lectures).*before (\d{1,2})(?: ?(am|pm))?",
            ],
            "max_lectures_per_day": [
                r"(max|maximum|no more than) (\d{1,2}) (class|lecture|lectures).*day",
            ],
            "min_break_between": [
                r"(at least|min|minumum) (\d{1,2}) (minute|min|hour|hrs).*break",
            ],
            "lab_slots": [
                r"(need|required|must have) (\d{1,2}) (lab|labs).*week",
            ],
            "faculty_availability": [
                r"(prof|professor|dr) (\w+).*free.*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            ],
            "preferred_subject_time": [
                r"(schedule|keep|put).*?(\w+).*?(morning|afternoon|evening)",
            ],
            "no_classes_on_day": [
                r"(no|avoid|dont|don’t).*(class|lecture|lectures).*on (monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            ],
        }

        # Days of week, time references
        self.days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        self.times = ["morning","afternoon","evening","night"]

    def _fuzzy_match(self, word, choices):
        """Return best fuzzy match for misspellings."""
        match, score, _ = process.extractOne(word, choices, scorer=fuzz.partial_ratio)
        return match if score > 80 else None

    def parse_text(self, text: str) -> dict:
        text = text.lower()
        constraints = {}

        # Try regex patterns
        for key, patterns in self.constraint_patterns.items():
            for pat in patterns:
                match = re.search(pat, text)
                if match:
                    if key in ["no_classes_after", "no_classes_before"]:
                        hour, meridian = match.groups()[-2], match.groups()[-1]
                        constraints[key] = f"{hour}{meridian or ''}"

                    elif key == "max_lectures_per_day":
                        num = match.group(2)
                        constraints[key] = int(num)

                    elif key == "min_break_between":
                        num, unit = match.group(2), match.group(3)
                        constraints[key] = f"{num} {unit}"

                    elif key == "lab_slots":
                        num = match.group(2)
                        constraints[key] = int(num)

                    elif key == "faculty_availability":
                        faculty = f"{match.group(1).title()} {match.group(2).title()}"
                        day = self._fuzzy_match(match.group(3), self.days)
                        constraints.setdefault("faculty_availability", {})[faculty] = day

                    elif key == "preferred_subject_time":
                        subject = match.group(2).title()
                        time = self._fuzzy_match(match.group(3), self.times)
                        constraints.setdefault("preferred_subject_time", {})[subject] = time

                    elif key == "no_classes_on_day":
                        day = match.group(3)
                        # Use setdefault to create a list if it doesn't exist, then append the day
                        constraints.setdefault("no_classes_on_day", []).append(day)

        return {"constraints": constraints}

    def parse_json(self, data: dict) -> dict:
        """Fallback for structured JSON input."""
        schema = {
            "branch": data.get("branch"),
            "courses": data.get("courses", []),
            "faculty_subjects": data.get("faculty_subjects", {}),
            "constraints": data.get("constraints", {})
        }
        return schema

    def parse(self, data):
        if isinstance(data, dict):
            return self.parse_json(data)
        elif isinstance(data, str):
            return self.parse_text(data)
        else:
            raise ValueError("Unsupported input type for NLP_Parser")


# Example usage
if __name__ == "__main__":
    parser = NLP_Parser()

    text_data = """
    No classes after 4pm.
    No lecures before 9 am.   # misspell handled
    Max 3 classes a day.
    At least 15 min break between.
    Need 2 labs per week.
    Professor Sharma only free on Monday.
    Schedule Maths in morning and Physics in evening.
    """

    print(parser.parse(text_data))
