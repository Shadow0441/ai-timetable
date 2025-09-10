import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from stable_baselines3 import PPO
import time
import logging

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constraint Programming Libraries ---
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    logger.warning("OR-Tools not installed. Constraint-based features will be unavailable.")
    ORTOOLS_AVAILABLE = False

# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================
@dataclass
class EnhancedConfig:
    # Core Timetable Constraints
    DAYS: int = 5
    HOURS_PER_DAY: int = 8
    LECTURES_PER_COURSE: int = 5
    MIN_LECTURES_PER_COURSE: int = 1

    # Solver & Hybrid Strategy
    HYBRID_MODE: bool = True
    FORCE_COMPLETE_SCHEDULE: bool = True
    MAX_COMPLETION_ATTEMPTS: int = 10
    RL_DRAFT_COMPLETION_TARGET: float = 0.7 # Target for how much of the schedule the RL model should draft

    # RL Model Training Parameters
    LEARNING_RATE: float = 0.0003
    DEMO_MODE: bool = True
    CP_TIMEOUT: int = 60
    INTENSIVE_TRAINING: int = 50000
    DEMO_TRAINING: int = 2000

    # Enhanced Rewards
    VALID_PLACEMENT: float = 100.0
    COMPLETION_BONUS: float = 10000.0
    SLOT_FILLING_BONUS: float = 50.0
    CONFLICT_PENALTY: float = -50.0
    FACULTY_CLASH_PENALTY: float = -100.0
    EMPTY_SLOT_PENALTY: float = -10.0
    # REFACTOR: Added OPTIMIZATION_BONUS for completing a course's lectures
    OPTIMIZATION_BONUS: float = 250.0

# =============================================================================
# COMPLETION-FOCUSED CONSTRAINT SOLVER
# =============================================================================
class ConstraintSolver:
    def __init__(self, config: EnhancedConfig):
        self.config = config

    # --- REFACTOR: New primary solving method that accepts an RL-generated draft ---
    def solve_with_rl_draft(self, courses: List[str], faculty: List[str], rooms: List[str],
                            course_faculty_map: Dict[str, str], rl_draft: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Solves the timetable problem using an RL-generated draft as a starting point or 'hint'.
        It will try to respect the draft while strictly enforcing all hard constraints.
        """
        if not ORTOOLS_AVAILABLE:
            return None, {'status': 'failed', 'reason': 'ortools_unavailable'}

        model = cp_model.CpModel()
        x = {} # Decision variables: x[(d, h, r, c)] is true if course c is in that slot

        num_courses = len(courses)
        num_rooms = len(rooms)

        for d in range(self.config.DAYS):
            for h in range(self.config.HOURS_PER_DAY):
                if h == 4: continue # Lunch break
                for r in range(num_rooms):
                    for c in range(num_courses):
                        x[(d, h, r, c)] = model.NewBoolVar(f'x_{d}_{h}_{r}_{c}')

        # --- Hard Constraints ---
        # 1. Each course must be scheduled for the required number of lectures.
        for c in range(num_courses):
            model.Add(sum(x[(d, h, r, c)] for d in range(self.config.DAYS)
                                          for h in range(self.config.HOURS_PER_DAY) if h != 4
                                          for r in range(num_rooms)) == self.config.LECTURES_PER_COURSE)

        # 2. A room can only have one course at a time.
        for d in range(self.config.DAYS):
            for h in range(self.config.HOURS_PER_DAY):
                if h == 4: continue
                for r in range(num_rooms):
                    model.Add(sum(x[(d, h, r, c)] for c in range(num_courses)) <= 1)

        # 3. A faculty member cannot teach two courses at the same time.
        faculty_courses = {fac: [] for fac in faculty}
        for c, course in enumerate(courses):
            fac = course_faculty_map.get(course)
            if fac:
                faculty_courses[fac].append(c)

        for fac, assigned_courses in faculty_courses.items():
            if len(assigned_courses) > 1:
                for d in range(self.config.DAYS):
                    for h in range(self.config.HOURS_PER_DAY):
                        if h == 4: continue
                        model.Add(sum(x[(d, h, r, c)] for r in range(num_rooms) for c in assigned_courses) <= 1)

        # --- Soft Constraint (Objective): Try to follow the RL draft ---
        objective_terms = []
        for d, h, r in np.ndindex(rl_draft.shape):
            if h == 4: continue
            draft_course_id = int(rl_draft[d, h, r])
            if draft_course_id > 0:
                draft_course_idx = draft_course_id - 1
                if draft_course_idx < num_courses:
                    # Add a positive term to the objective for matching the RL draft
                    objective_terms.append(x[(d, h, r, draft_course_idx)])

        model.Maximize(sum(objective_terms))

        # --- Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.CP_TIMEOUT
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            timetable = np.zeros((self.config.DAYS, self.config.HOURS_PER_DAY, num_rooms), dtype=np.int32)
            for d in range(self.config.DAYS):
                for h in range(self.config.HOURS_PER_DAY):
                    if h == 4: continue
                    for r in range(num_rooms):
                        for c in range(num_courses):
                            if solver.Value(x[(d, h, r, c)]) == 1:
                                timetable[d, h, r] = c + 1
            filled_slots = int(np.count_nonzero(timetable))
            stats = {'status': 'solved_with_rl_draft', 'solve_time': solver.WallTime(), 'filled_slots': filled_slots}
            return timetable, stats

        logger.warning("RL draft + CP failed. Falling back to greedy solver.")
        return self._solve_greedy_fill(courses, faculty, rooms, course_faculty_map)


    def _solve_greedy_fill(self, courses: List[str], faculty: List[str], rooms: List[str],
                           course_faculty_map: Dict[str, str]) -> Tuple[Optional[np.ndarray], Dict]:
        # Fallback greedy solver remains mostly the same
        logger.info("Using greedy fill strategy as a fallback")
        timetable = np.zeros((self.config.DAYS, self.config.HOURS_PER_DAY, len(rooms)), dtype=np.int32)
        lectures_remaining = {course: self.config.LECTURES_PER_COURSE for course in courses}

        available_slots = [(d, h, r) for d in range(self.config.DAYS) for h in range(self.config.HOURS_PER_DAY) if h != 4 for r in range(len(rooms))]
        random.shuffle(available_slots)

        faculty_schedule = {(d, h): set() for d in range(self.config.DAYS) for h in range(self.config.HOURS_PER_DAY)}

        for course in courses:
            for _ in range(lectures_remaining[course]):
                placed_in_slot = False
                for d, h, r in available_slots:
                     if timetable[d,h,r] == 0:
                        faculty_name = course_faculty_map.get(course, "Unknown")
                        if faculty_name not in faculty_schedule[(d, h)]:
                            course_idx = courses.index(course)
                            timetable[d, h, r] = course_idx + 1
                            faculty_schedule[(d, h)].add(faculty_name)
                            available_slots.remove((d,h,r))
                            placed_in_slot = True
                            break
                if not placed_in_slot:
                    logger.warning(f"Could not place all required lectures for {course} during greedy fill.")

        filled_slots = int(np.count_nonzero(timetable))
        required_slots = len(courses) * self.config.LECTURES_PER_COURSE
        status = 'solved' if filled_slots >= required_slots else 'partial'
        return timetable, {'status': f'greedy_{status}', 'filled_slots': filled_slots, 'required_slots': required_slots}

# =============================================================================
# RL ENVIRONMENT
# =============================================================================
class HybridTimetableEnv(gym.Env):
     def __init__(self, courses: List[str], faculty: List[str], rooms: List[str],
                 course_faculty_map: Dict[str, str], config: EnhancedConfig):
        super().__init__()
        self.config = config
        self.courses = courses
        self.faculty = faculty
        self.rooms = rooms
        self.course_faculty_map = course_faculty_map

        self.days = config.DAYS
        self.hours_per_day = config.HOURS_PER_DAY
        self.num_rooms = len(rooms)

        self.course_to_int = {course: i + 1 for i, course in enumerate(courses)}
        self.int_to_course = {i + 1: course for i, course in enumerate(courses)}
        self.faculty_to_int = {fac: i + 1 for i, fac in enumerate(faculty)}

        self.action_space = spaces.MultiDiscrete([self.days, self.hours_per_day, self.num_rooms])

        obs_size = (self.days * self.hours_per_day * self.num_rooms) + (len(courses) * 2) + 1
        self.observation_space = spaces.Box(low=0, high=len(courses) + 1, shape=(obs_size,), dtype=np.float32)

        self.reset()

     def _get_observation(self):
        flat_state = self.state.flatten()
        course_info = []
        total_needed = sum(self.lectures_needed.values())
        total_required = len(self.courses) * self.config.LECTURES_PER_COURSE
        total_placed = total_required - total_needed

        for course in self.courses:
            needed = self.lectures_needed.get(course, 0)
            placed = self.config.LECTURES_PER_COURSE - needed
            course_info.extend([needed, placed])

        completion_ratio = total_placed / total_required if total_required > 0 else 0
        course_info.append(completion_ratio)

        full_obs = np.concatenate([flat_state, course_info]).astype(np.float32)

        # Ensure observation matches the defined space size
        target_size = self.observation_space.shape[0]
        if len(full_obs) < target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(full_obs)] = full_obs
            return padded
        return full_obs[:target_size]

     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.zeros((self.days, self.hours_per_day, self.num_rooms), dtype=np.int32)
        # faculty_schedule now tracks faculty INT IDs for conflict checking
        self.faculty_schedule = np.zeros((self.days, self.hours_per_day), dtype=np.int32)
        self.lectures_needed = {course: self.config.LECTURES_PER_COURSE for course in self.courses}
        self.current_course_idx = 0
        self.steps_taken = 0
        self.max_steps = len(self.courses) * self.config.LECTURES_PER_COURSE * 2 # More realistic max steps
        return self._get_observation(), {}

     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        day, hour, room_idx = map(int, action)
        reward, terminated, truncated, info = 0.0, False, False, {}
        self.steps_taken += 1

        if self.steps_taken >= self.max_steps:
            truncated = True
            # Final reward calculation logic remains same
            total_needed = sum(self.lectures_needed.values())
            total_required = len(self.courses) * self.config.LECTURES_PER_COURSE
            completion_ratio = (total_required - total_needed) / total_required if total_required > 0 else 0
            reward = completion_ratio * self.config.COMPLETION_BONUS + self.config.EMPTY_SLOT_PENALTY * total_needed
            return self._get_observation(), reward, terminated, truncated, {"reason": "timeout"}

        available_courses = [c for c, needed in self.lectures_needed.items() if needed > 0]
        if not available_courses:
            terminated = True
            return self._get_observation(), self.config.COMPLETION_BONUS, terminated, truncated, {"reason": "completed"}

        course_to_place = available_courses[self.current_course_idx % len(available_courses)]
        course_int = self.course_to_int[course_to_place]
        faculty_name = self.course_faculty_map.get(course_to_place, "Unknown")
        faculty_int = self.faculty_to_int.get(faculty_name, 0)

        if hour == 4: # Penalize trying to schedule during lunch
             return self._get_observation(), self.config.CONFLICT_PENALTY * 0.5, terminated, truncated, {"reason": "lunch_break"}

        if self.state[day, hour, room_idx] != 0:
            reward = self.config.CONFLICT_PENALTY
        elif self.faculty_schedule[day, hour] != 0 and self.faculty_schedule[day, hour] != faculty_int:
            # More severe penalty for faculty clash
            reward = self.config.FACULTY_CLASH_PENALTY
        else:
            self.state[day, hour, room_idx] = course_int
            self.faculty_schedule[day, hour] = faculty_int
            self.lectures_needed[course_to_place] -= 1
            self.current_course_idx += 1
            reward = self.config.VALID_PLACEMENT + self.config.SLOT_FILLING_BONUS
            if self.lectures_needed[course_to_place] == 0:
                reward += self.config.OPTIMIZATION_BONUS

        return self._get_observation(), reward, terminated, truncated, info

# =============================================================================
# ORCHESTRATOR
# =============================================================================
class EnhancedTimetableDemo:
    def __init__(self, config: Optional[EnhancedConfig] = None):
        self.config = config if config else EnhancedConfig()
        self.constraint_solver = ConstraintSolver(self.config)

    def train_hybrid_model(self, env: HybridTimetableEnv):
        logger.info("🤖 Training hybrid AI model...")
        model = PPO("MlpPolicy", env, learning_rate=self.config.LEARNING_RATE, n_steps=1024, batch_size=64, n_epochs=10, verbose=0)
        steps = self.config.DEMO_TRAINING if self.config.DEMO_MODE else self.config.INTENSIVE_TRAINING
        logger.info(f"🎯 Training new model for {steps} timesteps...")
        model.learn(total_timesteps=steps)
        logger.info("✅ New model training complete.")
        return model

    # --- REFACTOR: This is the core logic change for a true hybrid approach ---
    def generate_hybrid_timetable(self, model, env: HybridTimetableEnv):
        logger.info("⚡ Generating timetable with TRUE HYBRID AI (RL Draft + CP Solve)...")

        # Step 1: Use the fine-tuned RL model to generate an intelligent "draft" schedule.
        # This draft contains placements that the model has learned are generally good,
        # but it may contain conflicts or be incomplete.
        logger.info("1/3: Generating intelligent draft with RL model...")
        rl_draft_timetable = np.zeros_like(env.state)
        obs, _ = env.reset()
        total_lectures_to_place = int(len(env.courses) * self.config.LECTURES_PER_COURSE * self.config.RL_DRAFT_COMPLETION_TARGET)

        for _ in range(total_lectures_to_place):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        rl_draft_timetable = env.state.copy() # Capture the state from the env
        logger.info("RL draft generated.")

        # Step 2: Pass the RL draft to the constraint solver. The solver will treat the
        # draft as a strong suggestion. It will fix any conflicts, schedule any
        # remaining lectures, and guarantee a 100% valid and complete timetable.
        logger.info("2/3: Solving constraints with RL draft as a hint...")
        final_timetable, stats = self.constraint_solver.solve_with_rl_draft(
            env.courses, env.faculty, env.rooms, env.course_faculty_map, rl_draft_timetable
        )

        if final_timetable is None:
             logger.error("All solving strategies failed, even with RL draft.")
             return None, {}

        # Step 3: Backfill any remaining empty slots with optional activities
        # to maximize the visual completeness and utility of the schedule.
        logger.info("3/3: Backfilling empty slots for maximum utilization...")
        final_timetable = self._backfill_timetable_intelligently(final_timetable, env)

        # --- Final Stats Calculation ---
        filled_slots = int(np.count_nonzero(final_timetable))
        required_slots = len(env.courses) * self.config.LECTURES_PER_COURSE
        total_available_slots = env.days * (env.hours_per_day - 1) * env.num_rooms

        # Using the filled slots from the CP solver for accuracy
        task_completion_rate = stats.get('filled_slots', 0) / required_slots if required_slots > 0 else 0
        utilization_rate = filled_slots / total_available_slots if total_available_slots > 0 else 0

        final_stats = {
            "filled_slots": stats.get('filled_slots', 0),
            "required_slots": required_slots,
            "total_available_slots": total_available_slots,
            "task_completion_rate": task_completion_rate,
            "utilization_rate": utilization_rate,
            "solver_stats": stats
        }
        logger.info(f"✅ Timetable generation finished. Completion: {stats.get('filled_slots',0)}/{required_slots} ({task_completion_rate:.2%}) | Utilization: {utilization_rate:.2%}")
        return final_timetable, final_stats


    def _backfill_timetable_intelligently(self, timetable: np.ndarray, env: HybridTimetableEnv) -> np.ndarray:
        logger.info("✍️ Backfilling empty slots with optional tutorial/study sessions...")

        faculty_to_courses = {}
        for course, faculty in env.course_faculty_map.items():
            faculty_to_courses.setdefault(faculty, []).append(course)

        faculty_schedule = {}
        for d, h, r in np.ndindex(timetable.shape):
            course_id = timetable[d, h, r]
            if course_id > 0 and course_id <= len(env.courses): # Only count core lectures
                course_name = env.int_to_course.get(course_id)
                faculty_name = env.course_faculty_map.get(course_name)
                if faculty_name:
                    faculty_schedule.setdefault((d, h), set()).add(faculty_name)

        empty_slots = []
        for d, h, r in np.ndindex(timetable.shape):
            if h != 4 and timetable[d, h, r] == 0:
                empty_slots.append((d, h, r))

        TUTORIAL_ID_OFFSET = 100 # To distinguish tutorials from core lectures
        slots_filled = 0
        for d, h, r in empty_slots:
            busy_faculty = faculty_schedule.get((d, h), set())
            free_faculty = [f for f in env.faculty if f not in busy_faculty]

            if free_faculty:
                faculty_to_assign = random.choice(free_faculty)
                if faculty_to_assign in faculty_to_courses:
                    # Assign a tutorial for one of the courses they teach
                    course_for_tutorial = random.choice(faculty_to_courses[faculty_to_assign])
                    course_id = env.course_to_int[course_for_tutorial]

                    timetable[d, h, r] = course_id + TUTORIAL_ID_OFFSET
                    faculty_schedule.setdefault((d, h), set()).add(faculty_to_assign)
                    slots_filled += 1

        logger.info(f"✅ Backfill complete. Added {slots_filled} optional sessions.")
        return timetable
