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
    """
    An advanced solver that uses Google's CP-SAT to generate a high-quality,
    realistic timetable, incorporating room types, resource limits, and workload balancing.
    """

    def __init__(self, config: Config, settings: 'InstituteSetting'):
        if not ORTOOLS_AVAILABLE:
            raise ImportError("Google OR-Tools is required to run the solver.")
        self.config = config
        self.settings = settings
        self.days_map = {day: i for i, day in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}

        start_time = datetime.strptime(self.settings.start_time, '%H:%M')
        end_time = datetime.strptime(self.settings.end_time, '%H:%M')
        self.hours_per_day = int((end_time - start_time).seconds / 3600)

        self.lunch_slot = None
        if self.settings.lunch_duration_hr > 0:
            self.lunch_slot = self.hours_per_day // 2 - 1


    def solve(self, subjects: List[str], faculty: List[str], rooms: List[str],
              faculty_map: Dict[str, str], num_lectures_map: Dict[str, int],
              nlp_constraints: Optional[List[Dict]] = None) -> Tuple[Optional[np.ndarray], Dict]:

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


        x = {} # Classroom assignments
        y = {} # Lab assignments

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

        # --- 3. Apply Hard Constraints (Fundamental Rules) ---
        self._apply_hard_constraints(model, x, y, locals())

        # --- 4. Apply Soft Constraints (For a "Good" Timetable) ---
        self._apply_soft_constraints(model, x, y, locals())

        # --- 5. Apply NLP Constraints (User Requests) ---
        if nlp_constraints:
            self._apply_nlp_constraints(model, x, y, nlp_constraints, locals())

        # --- 6. Solve the Model ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.CP_TIMEOUT
        status = solver.Solve(model)

        # --- 7. Process and Return the Solution ---
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # Combine classroom and lab results into one structure
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

            return timetable, {'status': 'solved', 'solve_time': solver.WallTime()}
        else:
            return None, {'status': 'failed', 'reason': 'No feasible solution found.'}

    def _apply_hard_constraints(self, model, x, y, var_maps):
        """Applies essential, unbreakable timetabling rules."""
        # Unpack necessary variables from the map for easier access
        g = lambda name: var_maps.get(name)

        # 1. Each subject is taught for its required number of lectures
        for s_name, s_idx in g('subject_map').items():
            total_lectures = []
            if s_name in g('lecture_subjects'):
                total_lectures.extend([x[d, h, r, s_idx] for d in range(self.config.DAYS) for h in range(self.hours_per_day) for r in range(g('num_classrooms'))])
            if s_name in g('lab_subjects'):
                 total_lectures.extend([y[d, h, l, s_idx] for d in range(self.config.DAYS) for h in range(self.hours_per_day) for l in range(g('num_labs'))])

            required_hours = g('num_lectures_map').get(s_name, self.config.DEFAULT_LECTURES_PER_COURSE)
            model.Add(sum(total_lectures) == required_hours)

        # 2. Resource Limit: At any time, number of classes cannot exceed available rooms
        for d in range(self.config.DAYS):
            for h in range(self.hours_per_day):
                # Total classrooms in use must be <= what the institute has
                model.Add(sum(x[d, h, r, s] for r in range(g('num_classrooms')) for s_name, s in g('subject_map').items() if s_name in g('lecture_subjects')) <= self.settings.total_rooms)
                # Total labs in use must be <= what the institute has
                model.Add(sum(y[d, h, l, s] for l in range(g('num_labs')) for s_name, s in g('subject_map').items() if s_name in g('lab_subjects')) <= self.settings.total_labs)

        # 3. A faculty member can only teach one class at a time, anywhere
        for d in range(self.config.DAYS):
            for h in range(self.hours_per_day):
                for f_idx in range(g('num_faculty')):
                    subjects_by_faculty = [g('subject_map')[s] for s, f in g('faculty_map').items() if f == list(g('faculty_map_indices').keys())[f_idx] and s in g('subject_map')]
                    if subjects_by_faculty:
                        lectures_in_classrooms = [x[d, h, r, s_idx] for r in range(g('num_classrooms')) for s_idx in subjects_by_faculty if g('subjects')[s_idx] in g('lecture_subjects')]
                        lectures_in_labs = [y[d, h, l, s_idx] for l in range(g('num_labs')) for s_idx in subjects_by_faculty if g('subjects')[s_idx] in g('lab_subjects')]
                        model.Add(sum(lectures_in_classrooms) + sum(lectures_in_labs) <= 1)

        # 4. No classes during lunch break
        if self.lunch_slot is not None:
            for d in range(self.config.DAYS):
                for s_idx in g('subject_map').values():
                    if g('subjects')[s_idx] in g('lecture_subjects'):
                        for r in range(g('num_classrooms')): model.Add(x[d, self.lunch_slot, r, s_idx] == 0)
                    if g('subjects')[s_idx] in g('lab_subjects'):
                        for l in range(g('num_labs')): model.Add(y[d, self.lunch_slot, l, s_idx] == 0)

    def _apply_soft_constraints(self, model, x, y, var_maps):
        """Adds constraints to improve the quality and feasibility of the timetable."""
        g = lambda name: var_maps.get(name)
        total_lectures = sum(g('num_lectures_map').values())
        min_lectures_per_day = (total_lectures // self.config.DAYS) - 1 # Allow some flexibility

        # 1. Encourage a balanced number of lectures each day
        for d in range(self.config.DAYS):
            lectures_today_in_classrooms = [x[d, h, r, s] for h in range(self.hours_per_day) for r in range(g('num_classrooms')) for s_name, s in g('subject_map').items() if s_name in g('lecture_subjects')]
            lectures_today_in_labs = [y[d, h, l, s] for h in range(self.hours_per_day) for l in range(g('num_labs')) for s_name, s in g('subject_map').items() if s_name in g('lab_subjects')]
            model.Add(sum(lectures_today_in_classrooms) + sum(lectures_today_in_labs) >= min_lectures_per_day)


    def _apply_nlp_constraints(self, model, x, y, constraints, var_maps):
        """Translates parsed NLP constraints into model constraints."""
        g = lambda name: var_maps.get(name)

        for const in constraints:
            ctype = const.get('type')
            target_subject = const.get('target') or const.get('subject')
            if not target_subject or target_subject not in g('subject_map'): continue

            s_idx = g('subject_map')[target_subject]

            if ctype == 'room_assignment':
                room = const.get('room')
                # Determine if it's a classroom or lab
                if room in g('classroom_map'):
                    r_idx = g('classroom_map')[room]
                    # This subject must only happen in this classroom
                    for d in range(self.config.DAYS):
                        for h in range(self.hours_per_day):
                           model.Add(sum(x[d, h, r, s_idx] for r in range(g('num_classrooms')) if r != r_idx) == 0)
                elif room in g('lab_map'):
                    l_idx = g('lab_map')[room]
                    # This subject must only happen in this lab
                    for d in range(self.config.DAYS):
                        for h in range(self.hours_per_day):
                           model.Add(sum(y[d, h, l, s_idx] for l in range(g('num_labs')) if l != l_idx) == 0)

            # (Add logic for other NLP constraint types here, similar to the original solver)

