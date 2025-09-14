# nlp_parser.py
import spacy
from spacy.matcher import PhraseMatcher
import logging

logger = logging.getLogger(__name__)

class NLPParser:
    """
    Parses natural language text to extract scheduling constraints using spaCy.
    """
    def __init__(self, courses, faculty, rooms):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Create patterns for all custom entities for efficient matching
        course_patterns = [self.nlp.make_doc(c) for c in courses]
        faculty_patterns = [self.nlp.make_doc(f) for f in faculty]
        room_patterns = [self.nlp.make_doc(r) for r in rooms]

        self.matcher.add("COURSE", course_patterns)
        self.matcher.add("FACULTY", faculty_patterns)
        self.matcher.add("ROOM", room_patterns)

        # Simple mapping for time and day parsing
        self.day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4
        }
        self.time_map = {
            "morning": [0, 1, 2, 3], # 9am-12pm
            "afternoon": [5, 6, 7]  # 2pm-4pm
        }


    def parse(self, text: str) -> list[dict]:
        """
        Processes a string of text and returns a list of structured constraint dictionaries.
        """
        if not text:
            return []

        doc = self.nlp(text.lower())
        matches = self.matcher(doc)

        # --- Intent & Entity Extraction ---
        constraint = {}
        # Simple intent recognition
        if "must" in doc.text or "has to" in doc.text or "needs to be" in doc.text:
            constraint['type'] = 'requirement' # This is a hard rule
        else:
            constraint['type'] = 'preference'  # This is a soft preference

        # Extract entities found by the PhraseMatcher
        for match_id, start, end in matches:
            entity_type = self.nlp.vocab.strings[match_id]
            entity_text = doc[start:end].text
            constraint[entity_type.lower()] = entity_text.title() # Normalize to title case

        # Extract generic day/time entities
        for word in doc:
            if word.lemma_ in self.day_map:
                constraint['day'] = self.day_map[word.lemma_]
            if word.lemma_ in self.time_map:
                constraint['hours'] = self.time_map[word.lemma_]

        if len(constraint) > 1: # A valid constraint must have more than just a 'type'
            logger.info(f"NLP Parser extracted constraint: {constraint}")
            return [constraint] # Returning a list to support multiple constraints in the future

        return []
