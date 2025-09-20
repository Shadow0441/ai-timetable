import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    logging.warning("OR-Tools not installed. CP-solver features will be unavailable.")
    ORTOOLS_AVAILABLE = False

from config import Config

class ConstraintSolver:


    def __init__(self, config: Config, settings: 'InstituteSetting'):
        if not ORTOOLS_AVAILABLE:
            raise ImportError("Google OR-Tools is required to run the solver.")

        self.config = config
        self.settings = settings
        self.days_map = {day: i for i, day in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
        self.time_map = {"morning": (0, 4), "afternoon": (4, 8), "evening": (8, 12), "night": (12, 24)}

        start_time = datetime.strptime(self.settings.start_time, '%H:%M')
        end_time = datetime.strptime(self.settings.end_time, '%H:%M')
        self.hours_per_day = int((end_time - start_time).seconds / 3600)

        self.lunch_slot = None
        if self.settings.lunch_duration_hr > 0:
            self.lunch_slot = self.hours_per_day // 2 - 1

    def solve(self, subjects: List[str], faculty: List[str], rooms: List[str],
              faculty_map: Dict[str, str], num_lectures_map: Dict[str, int],
              nlp_constraints: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Dict]:

        model = cp_model.CpModel()

        # --- 1. Intelligent Categorization of Rooms and Subjects ---
        classrooms = [r for r in rooms if 'lab' not in r.lower()]
        labs = [r for r in rooms if 'lab' in r.lower()]

        lecture_subjects = [s for s in subjects if 'lab' not in s.lower() and 'workshop' not in s.lower()]
        lab_subjects = [s for s in subjects if 'lab' in s.lower() or 'workshop' in s.lower()]

        # Create maps for efficient index lookup
        subject_map = {name: i for i, name in enumerate(subjects)}
        classroom_map = {name: i for i, name in enumerate(classrooms)}
        lab_map = {name: i for i, name in enumerate(labs)}
        faculty_map_indices = {name: i for i, name in enumerate(faculty)}

        num_classrooms = len(classrooms)
        num_labs = len(labs)
        num_subjects = len(subjects)
        num_faculty = len(faculty)

        x = {}  # Classroom assignments
        y = {}  # Lab assignments

        for d in range(self.config.DAYS):
            for h in range(self.hours_per_day):
                for s_name in subjects:
                    s_idx = subject_map[s_name]
                    # Create variables for classrooms only if it's a lecture subject
                    if s_name in lecture_subjects:
                        for r_idx in range(num_classrooms):
                             x[d, h, r_idx, s_idx] = model.NewBoolVar(f'x_{d}_{h}_{r_idx}_{s_idx}')
                    # Create variables for labs only if it's a lab subject
                    if s_name in lab_subjects:
                        for l_idx in range(num_labs):
                            y[d, h, l_idx, s_idx] = model.NewBoolVar(f'y_{d}_{h}_{l_idx}_{s_idx}')

        # --- 2. Apply Hard Constraints (Fundamental Rules) ---
        self._apply_hard_constraints(model, x, y, locals())

        # --- 3. Apply Soft Constraints (For a "Good" Timetable) ---
        self._apply_soft_constraints(model, x, y, locals())

        # --- 4. Apply NLP Constraints (User Requests) ---
        if nlp_constraints and 'constraints' in nlp_constraints:
            self._apply_nlp_constraints(model, x, y, nlp_constraints['constraints'], locals())

        # --- 5. Solve the Model ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.CP_TIMEOUT
        status = solver.Solve(model)

        # --- 6. Process and Return the Solution ---
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Enhanced room naming - use "Class" for classrooms and "Lab" for labs
            enhanced_rooms = []
            for room in rooms:
                if 'lab' in room.lower():
                    enhanced_rooms.append("Lab")
                else:
                    enhanced_rooms.append("Class")

            num_total_rooms = len(rooms)
            timetable = np.zeros((self.config.DAYS, self.hours_per_day, num_total_rooms), dtype=int)

            for d in range(self.config.DAYS):
                for h in range(self.hours_per_day):
                    # Populate classroom lectures
                    for r_idx, r_name in enumerate(classrooms):
                        final_r_idx = rooms.index(r_name)
                        for s_idx, s_name in enumerate(lecture_subjects):
                            s_final_idx = subject_map[s_name]
                            if solver.Value(x[d, h, r_idx, s_final_idx]) == 1:
                                timetable[d, h, final_r_idx] = s_final_idx + 1
                    # Populate lab lectures
                    for l_idx, l_name in enumerate(labs):
                        final_l_idx = rooms.index(l_name)
                        for s_idx, s_name in enumerate(lab_subjects):
                            s_final_idx = subject_map[s_name]
                            if solver.Value(y[d, h, l_idx, s_final_idx]) == 1:
                                timetable[d, h, final_l_idx] = s_final_idx + 1

            # Return enhanced room names for display
            return timetable, {
                'status': 'solved',
                'solve_time': solver.WallTime(),
                'room_names': enhanced_rooms  # Enhanced room names
            }
        else:
            return None, {'status': 'failed', 'reason': 'No feasible solution found.'}

    def _apply_hard_constraints(self, model, x, y, var_maps):
        """Applies essential, unbreakable timetabling rules."""
        g = lambda name: var_maps.get(name)

        # 1. Each subject is taught for its required number of lectures
        for s_name, s_idx in g('subject_map').items():
            total_lectures = []
            if s_name in g('lecture_subjects'):
                total_lectures.extend([x[d, h, r, s_idx] for d in range(self.config.DAYS)
                                     for h in range(self.hours_per_day) for r in range(g('num_classrooms'))])
            if s_name in g('lab_subjects'):
                 total_lectures.extend([y[d, h, l, s_idx] for d in range(self.config.DAYS)
                                      for h in range(self.hours_per_day) for l in range(g('num_labs'))])

            required_hours = g('num_lectures_map').get(s_name, self.config.DEFAULT_LECTURES_PER_COURSE)
            model.Add(sum(total_lectures) == required_hours)

        # 2. Resource Limit: At any time, number of classes cannot exceed available rooms
        for d in range(self.config.DAYS):
            for h in range(self.hours_per_day):
                # Total classrooms in use must be <= what the institute has
                classroom_lectures = [x[d, h, r, s] for r in range(g('num_classrooms'))
                                    for s_name, s in g('subject_map').items() if s_name in g('lecture_subjects')]
                if classroom_lectures:
                    model.Add(sum(classroom_lectures) <= self.settings.total_rooms)

                # Total labs in use must be <= what the institute has
                lab_lectures = [y[d, h, l, s] for l in range(g('num_labs'))
                              for s_name, s in g('subject_map').items() if s_name in g('lab_subjects')]
                if lab_lectures:
                    model.Add(sum(lab_lectures) <= self.settings.total_labs)

        # 3. A faculty member can only teach one class at a time
        for d in range(self.config.DAYS):
            for h in range(self.hours_per_day):
                for f_name in g('faculty'):
                    f_idx = g('faculty_map_indices')[f_name]
                    subjects_by_faculty = [g('subject_map')[s] for s, f in g('faculty_map').items()
                                         if f == f_name and s in g('subject_map')]
                    if subjects_by_faculty:
                        lectures_in_classrooms = [x[d, h, r, s_idx] for r in range(g('num_classrooms'))
                                                for s_idx in subjects_by_faculty
                                                if g('subjects')[s_idx] in g('lecture_subjects')]
                        lectures_in_labs = [y[d, h, l, s_idx] for l in range(g('num_labs'))
                                          for s_idx in subjects_by_faculty
                                          if g('subjects')[s_idx] in g('lab_subjects')]
                        all_lectures = lectures_in_classrooms + lectures_in_labs
                        if all_lectures:
                            model.Add(sum(all_lectures) <= 1)

        # 4. One room can only host one subject at a time
        for d in range(self.config.DAYS):
            for h in range(self.hours_per_day):
                # For classrooms
                for r in range(g('num_classrooms')):
                    classroom_subjects = [x[d, h, r, s] for s_name, s in g('subject_map').items()
                                        if s_name in g('lecture_subjects')]
                    if classroom_subjects:
                        model.Add(sum(classroom_subjects) <= 1)

                # For labs
                for l in range(g('num_labs')):
                    lab_subjects = [y[d, h, l, s] for s_name, s in g('subject_map').items()
                                  if s_name in g('lab_subjects')]
                    if lab_subjects:
                        model.Add(sum(lab_subjects) <= 1)

        # 5. No classes during lunch break
        if self.lunch_slot is not None:
            for d in range(self.config.DAYS):
                for s_idx in g('subject_map').values():
                    s_name = g('subjects')[s_idx]
                    if s_name in g('lecture_subjects'):
                        for r in range(g('num_classrooms')):
                            model.Add(x[d, self.lunch_slot, r, s_idx] == 0)
                    if s_name in g('lab_subjects'):
                        for l in range(g('num_labs')):
                            model.Add(y[d, self.lunch_slot, l, s_idx] == 0)

    def _apply_soft_constraints(self, model, x, y, var_maps):
        """Adds constraints to improve the quality and feasibility of the timetable."""
        g = lambda name: var_maps.get(name)
        total_lectures = sum(g('num_lectures_map').values())
        min_lectures_per_day = max(1, (total_lectures // self.config.DAYS) - 1)  # Allow some flexibility

        # 1. Encourage a balanced number of lectures each day
        for d in range(self.config.DAYS):
            lectures_today_in_classrooms = [x[d, h, r, s] for h in range(self.hours_per_day)
                                          for r in range(g('num_classrooms'))
                                          for s_name, s in g('subject_map').items()
                                          if s_name in g('lecture_subjects')]
            lectures_today_in_labs = [y[d, h, l, s] for h in range(self.hours_per_day)
                                    for l in range(g('num_labs'))
                                    for s_name, s in g('subject_map').items()
                                    if s_name in g('lab_subjects')]
            all_lectures_today = lectures_today_in_classrooms + lectures_today_in_labs
            if all_lectures_today:
                model.Add(sum(all_lectures_today) >= min_lectures_per_day)

        # 2. Avoid consecutive identical subjects (spread subjects across the week)
        for s_name, s_idx in g('subject_map').items():
            for d in range(self.config.DAYS):
                for h in range(self.hours_per_day - 1):  # Check consecutive hours
                    current_hour_lectures = []
                    next_hour_lectures = []

                    if s_name in g('lecture_subjects'):
                        current_hour_lectures.extend([x[d, h, r, s_idx] for r in range(g('num_classrooms'))])
                        next_hour_lectures.extend([x[d, h+1, r, s_idx] for r in range(g('num_classrooms'))])
                    if s_name in g('lab_subjects'):
                        current_hour_lectures.extend([y[d, h, l, s_idx] for l in range(g('num_labs'))])
                        next_hour_lectures.extend([y[d, h+1, l, s_idx] for l in range(g('num_labs'))])

                    if current_hour_lectures and next_hour_lectures:
                        # Soft constraint: try to avoid back-to-back same subjects
                        model.Add(sum(current_hour_lectures) + sum(next_hour_lectures) <= 1)

    def _apply_nlp_constraints(self, model, x, y, constraints, var_maps):
        """Translates parsed NLP constraints into model constraints."""
        g = lambda name: var_maps.get(name)

        # Handle "no_classes_after" constraint
        if "no_classes_after" in constraints:
            time_limit = constraints["no_classes_after"]
            hour_limit = self._parse_time_to_hour(time_limit)
            if hour_limit is not None:
                for d in range(self.config.DAYS):
                    for h in range(hour_limit, self.hours_per_day):
                        for s_idx in g('subject_map').values():
                            s_name = g('subjects')[s_idx]
                            if s_name in g('lecture_subjects'):
                                for r in range(g('num_classrooms')):
                                    model.Add(x[d, h, r, s_idx] == 0)
                            if s_name in g('lab_subjects'):
                                for l in range(g('num_labs')):
                                    model.Add(y[d, h, l, s_idx] == 0)

        # Handle "no_classes_before" constraint
        if "no_classes_before" in constraints:
            time_limit = constraints["no_classes_before"]
            hour_limit = self._parse_time_to_hour(time_limit)
            if hour_limit is not None:
                for d in range(self.config.DAYS):
                    for h in range(0, hour_limit):
                        for s_idx in g('subject_map').values():
                            s_name = g('subjects')[s_idx]
                            if s_name in g('lecture_subjects'):
                                for r in range(g('num_classrooms')):
                                    model.Add(x[d, h, r, s_idx] == 0)
                            if s_name in g('lab_subjects'):
                                for l in range(g('num_labs')):
                                    model.Add(y[d, h, l, s_idx] == 0)

        if "no_classes_on_day" in constraints:
            forbidden_days = constraints["no_classes_on_day"]
            for day_name in forbidden_days:
                if day_name in self.days_map:
                    day_idx = self.days_map[day_name]
                    # Forbid all classes on this day
                    for h in range(self.hours_per_day):
                        for s_idx in g('subject_map').values():
                            s_name = g('subjects')[s_idx]
                            if s_name in g('lecture_subjects'):
                                for r in range(g('num_classrooms')):
                                    model.Add(x[day_idx, h, r, s_idx] == 0)
                            if s_name in g('lab_subjects'):
                                for l in range(g('num_labs')):
                                    model.Add(y[day_idx, h, l, s_idx] == 0)
        # Handle "max_lectures_per_day" constraint
        if "max_lectures_per_day" in constraints:
            max_lectures = constraints["max_lectures_per_day"]
            for d in range(self.config.DAYS):
                all_lectures_today = []
                for s_idx in g('subject_map').values():
                    s_name = g('subjects')[s_idx]
                    if s_name in g('lecture_subjects'):
                        all_lectures_today.extend([x[d, h, r, s_idx] for h in range(self.hours_per_day)
                                                 for r in range(g('num_classrooms'))])
                    if s_name in g('lab_subjects'):
                        all_lectures_today.extend([y[d, h, l, s_idx] for h in range(self.hours_per_day)
                                                 for l in range(g('num_labs'))])
                if all_lectures_today:
                    model.Add(sum(all_lectures_today) <= max_lectures)

        # Handle "min_break_between" constraint
        if "min_break_between" in constraints:
            min_break_str = constraints["min_break_between"]
            break_hours = self._parse_break_duration(min_break_str)
            if break_hours > 0:
                for d in range(self.config.DAYS):
                    for h in range(self.hours_per_day - break_hours):
                        for s_idx in g('subject_map').values():
                            s_name = g('subjects')[s_idx]
                            current_lectures = []
                            future_lectures = []

                            if s_name in g('lecture_subjects'):
                                current_lectures.extend([x[d, h, r, s_idx] for r in range(g('num_classrooms'))])
                                future_lectures.extend([x[d, h + break_hours, r, s_idx]
                                                      for r in range(g('num_classrooms'))])
                            if s_name in g('lab_subjects'):
                                current_lectures.extend([y[d, h, l, s_idx] for l in range(g('num_labs'))])
                                future_lectures.extend([y[d, h + break_hours, l, s_idx]
                                                      for l in range(g('num_labs'))])

                            if current_lectures and future_lectures:
                                # If a subject is scheduled now, it can't be scheduled too soon
                                model.Add(sum(current_lectures) + sum(future_lectures) <= 1)

        # Handle "lab_slots" constraint
        if "lab_slots" in constraints:
            required_lab_slots = constraints["lab_slots"]
            for s_name in g('lab_subjects'):
                s_idx = g('subject_map')[s_name]
                total_lab_lectures = [y[d, h, l, s_idx] for d in range(self.config.DAYS)
                                    for h in range(self.hours_per_day) for l in range(g('num_labs'))]
                if total_lab_lectures:
                    model.Add(sum(total_lab_lectures) >= required_lab_slots)

        # Handle "faculty_availability" constraint
        if "faculty_availability" in constraints:
            faculty_constraints = constraints["faculty_availability"]
            for faculty_name, available_day in faculty_constraints.items():
                if available_day in self.days_map and faculty_name in g('faculty'):
                    available_day_idx = self.days_map[available_day.lower()]
                    # Find subjects taught by this faculty
                    faculty_subjects = [g('subject_map')[s] for s, f in g('faculty_map').items()
                                      if f == faculty_name and s in g('subject_map')]

                    # Restrict this faculty to only teach on their available day
                    for d in range(self.config.DAYS):
                        if d != available_day_idx:  # Not the available day
                            for s_idx in faculty_subjects:
                                s_name = g('subjects')[s_idx]
                                if s_name in g('lecture_subjects'):
                                    for h in range(self.hours_per_day):
                                        for r in range(g('num_classrooms')):
                                            model.Add(x[d, h, r, s_idx] == 0)
                                if s_name in g('lab_subjects'):
                                    for h in range(self.hours_per_day):
                                        for l in range(g('num_labs')):
                                            model.Add(y[d, h, l, s_idx] == 0)

        # Handle "preferred_subject_time" constraint
        if "preferred_subject_time" in constraints:
            time_preferences = constraints["preferred_subject_time"]
            for subject_name, preferred_time in time_preferences.items():
                if subject_name.lower() in [s.lower() for s in g('subjects')] and preferred_time in self.time_map:
                    # Find the actual subject name (case-insensitive match)
                    actual_subject = next((s for s in g('subjects') if s.lower() == subject_name.lower()), None)
                    if actual_subject:
                        s_idx = g('subject_map')[actual_subject]
                        start_hour, end_hour = self.time_map[preferred_time]
                        # Convert to actual schedule hours
                        start_hour = max(0, start_hour - 9)  # Assuming 9 AM start
                        end_hour = min(self.hours_per_day, end_hour - 9)

                        if start_hour < end_hour:
                            # Schedule this subject only in preferred time slots
                            for d in range(self.config.DAYS):
                                # Forbidden hours (outside preferred time)
                                forbidden_hours = list(range(0, start_hour)) + list(range(end_hour, self.hours_per_day))
                                for h in forbidden_hours:
                                    if actual_subject in g('lecture_subjects'):
                                        for r in range(g('num_classrooms')):
                                            model.Add(x[d, h, r, s_idx] == 0)
                                    if actual_subject in g('lab_subjects'):
                                        for l in range(g('num_labs')):
                                            model.Add(y[d, h, l, s_idx] == 0)

    def _parse_time_to_hour(self, time_str):
        """Convert time string like '4pm' to hour index in schedule."""
        try:
            import re
            match = re.search(r'(\d{1,2})\s*(am|pm)?', time_str.lower())
            if match:
                hour = int(match.group(1))
                meridian = match.group(2)

                if meridian == 'pm' and hour != 12:
                    hour += 12
                elif meridian == 'am' and hour == 12:
                    hour = 0

                # Convert to schedule hour (assuming 9 AM start)
                start_hour = int(self.settings.start_time.split(':')[0])
                return hour - start_hour
        except:
            pass
        return None

    def _parse_break_duration(self, break_str):
        """Convert break duration string to hours."""
        try:
            import re
            match = re.search(r'(\d+)\s*(min|minute|hour|hrs|hr)', break_str.lower())
            if match:
                value = int(match.group(1))
                unit = match.group(2)

                if 'min' in unit:
                    return max(1, value // 60)  # Convert minutes to hours, minimum 1 hour
                elif 'hour' in unit or 'hr' in unit:
                    return value
        except:
            pass
        return 1  # Default minimum break of 1 hour
