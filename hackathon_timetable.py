import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import time
import logging
from tabulate import tabulate

from nlp_parser import NLPParser

# --- Application Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    logger.warning("OR-Tools not installed. CP-solver features will be unavailable.")
    ORTOOLS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class EnhancedConfig:
    # File Paths
    BACKUP_MODEL_PATH: str = "instance/pretrained_backup_model.zip"

    # Timetable Structure
    DAYS: int = 5
    HOURS_PER_DAY: int = 8
    LECTURES_PER_COURSE: int = 5

    # Solver Strategy
    CP_TIMEOUT: int = 60
    RL_DRAFT_COMPLETION_TARGET: float = 0.7

    # RL Training
    LEARNING_RATE: float = 0.0003
    DEMO_MODE: bool = True
    INTENSIVE_TRAINING: int = 50000
    DEMO_TRAINING: int = 2000

    # Rewards & Penalties
    VALID_PLACEMENT: float = 100.0
    COMPLETION_BONUS: float = 10000.0
    SLOT_FILLING_BONUS: float = 50.0
    OPTIMIZATION_BONUS: float = 250.0
    CONFLICT_PENALTY: float = -50.0
    FACULTY_CLASH_PENALTY: float = -100.0
    EMPTY_SLOT_PENALTY: float = -10.0
    NLP_PREFERENCE_BONUS: float = 500.0
    NLP_REQUIREMENT_PENALTY: float = -2000.0

# =============================================================================
# RL ENVIRONMENT
# =============================================================================
class HybridTimetableEnv(gym.Env):
    """Custom Gym environment for the timetable scheduling problem."""
    def __init__(self, courses, faculty, rooms, course_faculty_map, config, custom_constraints: Optional[List[Dict]] = None):
        super().__init__()
        self.config, self.courses, self.faculty, self.rooms, self.course_faculty_map = config, courses, faculty, rooms, course_faculty_map
        self.custom_constraints = custom_constraints if custom_constraints else []
        self.days, self.hours_per_day, self.num_rooms = config.DAYS, config.HOURS_PER_DAY, len(rooms)

        self.course_to_int = {c: i + 1 for i, c in enumerate(courses)}
        self.int_to_course = {i + 1: c for i, c in enumerate(courses)}
        self.faculty_to_int = {f: i + 1 for i, f in enumerate(faculty)}

        self.action_space = spaces.MultiDiscrete([self.days, self.hours_per_day, self.num_rooms])
        obs_size = (self.days * self.hours_per_day * self.num_rooms) + (len(courses) * 2) + 1
        self.observation_space = spaces.Box(low=0, high=len(courses) + 1, shape=(obs_size,), dtype=np.float32)
        self.reset()

    def _get_observation(self):
        flat_state = self.state.flatten()
        course_info = []
        total_required = len(self.courses) * self.config.LECTURES_PER_COURSE
        total_placed = total_required - sum(self.lectures_needed.values())

        for course in self.courses:
            needed = self.lectures_needed.get(course, 0)
            course_info.extend([needed, self.config.LECTURES_PER_COURSE - needed])

        completion_ratio = total_placed / total_required if total_required > 0 else 0
        course_info.append(completion_ratio)

        full_obs = np.concatenate([flat_state, course_info]).astype(np.float32)

        # Pad or truncate observation to match the defined space
        target_size = self.observation_space.shape[0]
        if len(full_obs) < target_size:
            padded = np.zeros(target_size)
            padded[:len(full_obs)] = full_obs
            return padded
        return full_obs[:target_size]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'custom_constraints' in options:
            self.custom_constraints = options['custom_constraints']

        self.state = np.zeros((self.days, self.hours_per_day, self.num_rooms), dtype=np.int32)
        self.faculty_schedule = {} # Using dict for sparse tracking
        self.lectures_needed = {c: self.config.LECTURES_PER_COURSE for c in self.courses}
        self.current_course_idx = 0
        self.steps_taken = 0
        self.max_steps = len(self.courses) * self.config.LECTURES_PER_COURSE * 2 # Safety break
        return self._get_observation(), {}

    def step(self, action):
        day, hour, room_idx = map(int, action)
        reward, terminated, truncated = 0.0, False, False
        self.steps_taken += 1

        if self.steps_taken >= self.max_steps:
            truncated = True
            return self._get_observation(), 0, terminated, truncated, {"reason": "timeout"}

        available_courses = [c for c, n in self.lectures_needed.items() if n > 0]
        if not available_courses:
            terminated = True
            return self._get_observation(), self.config.COMPLETION_BONUS, terminated, truncated, {"reason": "completed"}

        course = available_courses[self.current_course_idx % len(available_courses)]
        c_int = self.course_to_int[course]
        f_name = self.course_faculty_map.get(course)

        # Apply NLP-based reward shaping
        for const in self.custom_constraints:
            is_relevant = const.get('course') == course or const.get('faculty') == f_name
            if is_relevant:
                day_match = const.get('day') is None or const['day'] == day
                hour_match = const.get('hours') is None or hour in const['hours']
                if day_match and hour_match:
                    reward += self.config.NLP_PREFERENCE_BONUS
                elif const['type'] == 'requirement':
                    reward += self.config.NLP_REQUIREMENT_PENALTY

        # Apply environment rules and penalties
        if hour == 4: # Lunch break
            return self._get_observation(), self.config.CONFLICT_PENALTY, terminated, truncated, {"reason": "lunch"}
        if self.state[day, hour, room_idx] != 0:
            reward += self.config.CONFLICT_PENALTY
        elif f_name in self.faculty_schedule.get((day, hour), set()):
            reward += self.config.FACULTY_CLASH_PENALTY
        else: # Valid placement
            self.state[day, hour, room_idx] = c_int
            self.faculty_schedule.setdefault((day, hour), set()).add(f_name)
            self.lectures_needed[course] -= 1
            self.current_course_idx += 1
            reward += self.config.VALID_PLACEMENT
            if self.lectures_needed[course] == 0:
                reward += self.config.OPTIMIZATION_BONUS

        return self._get_observation(), reward, terminated, truncated, {}


# =============================================================================
# CONSTRAINT SOLVER
# =============================================================================
class ConstraintSolver:
    """Uses Google OR-Tools to solve or optimize the timetable."""
    def __init__(self, config: EnhancedConfig):
        self.config = config

    def solve_with_rl_draft(self, courses, faculty, rooms, course_faculty_map, rl_draft, custom_constraints: list):
        if not ORTOOLS_AVAILABLE: return rl_draft, {'status': 'ortools_unavailable'}

        model = cp_model.CpModel()
        num_courses, num_rooms = len(courses), len(rooms)
        course_to_int = {c: i for i, c in enumerate(courses)}

        # Define variables
        x = {}
        for d, h, r, c in np.ndindex(self.config.DAYS, self.config.HOURS_PER_DAY, num_rooms, num_courses):
            if h != 4: x[(d, h, r, c)] = model.NewBoolVar(f'x_{d}_{h}_{r}_{c}')

        # Hard constraints
        for c in range(num_courses):
            model.Add(sum(x.get((d, h, r, c), 0) for d, h, r in np.ndindex(self.config.DAYS, self.config.HOURS_PER_DAY, num_rooms) if h != 4) == self.config.LECTURES_PER_COURSE)
        for d, h, r in np.ndindex(self.config.DAYS, self.config.HOURS_PER_DAY, num_rooms):
            if h != 4: model.Add(sum(x.get((d, h, r, c), 0) for c in range(num_courses)) <= 1)

        faculty_courses = {f: [course_to_int[c] for c, fac in course_faculty_map.items() if fac == f] for f in faculty}
        for c_indices in faculty_courses.values():
            if len(c_indices) > 1:
                for d, h in np.ndindex(self.config.DAYS, self.config.HOURS_PER_DAY):
                    if h != 4: model.Add(sum(x.get((d, h, r, c), 0) for r in range(num_rooms) for c in c_indices) <= 1)

        # Apply NLP requirement constraints
        for const in custom_constraints:
            if const.get('type') == 'requirement':
                c_idx = course_to_int.get(const.get('course'))
                r_idx = rooms.index(const['room']) if const.get('room') in rooms else None
                if c_idx is not None and r_idx is not None:
                     model.Add(sum(x.get((d, h, r_idx, c_idx), 0) for d, h in np.ndindex(self.config.DAYS, self.config.HOURS_PER_DAY) if h != 4) > 0)

        # Objective: Maximize agreement with RL draft
        objective_terms = []
        for d, h, r in np.ndindex(rl_draft.shape):
            if h != 4 and rl_draft[d, h, r] > 0:
                c_idx = int(rl_draft[d, h, r]) - 1
                if 0 <= c_idx < num_courses:
                    objective_terms.append(x[(d, h, r, c_idx)])
        model.Maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.CP_TIMEOUT
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            timetable = np.zeros_like(rl_draft)
            for (d, h, r, c), var in x.items():
                if solver.Value(var) == 1:
                    timetable[d, h, r] = c + 1
            return timetable, {'status': 'cp_solved', 'filled_slots': int(np.count_nonzero(timetable))}
        return rl_draft, {'status': 'cp_failed', 'filled_slots': int(np.count_nonzero(rl_draft))}

# =============================================================================
# TRAINING AND ORCHESTRATION
# =============================================================================
def train_and_save_model(courses, faculty, rooms, course_faculty_map, save_path):
    """Trains a PPO model on a specific dataset and saves it."""
    logger.info(f"Starting custom model training for data with {len(courses)} courses.")
    config = EnhancedConfig(DEMO_MODE=True)
    env = DummyVecEnv([lambda: HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, config)])

    model = PPO("MlpPolicy", env, learning_rate=config.LEARNING_RATE, verbose=0)
    training_steps = max(config.DEMO_TRAINING, len(courses) * config.LECTURES_PER_COURSE * 100)

    logger.info(f"Training for {training_steps} timesteps...")
    model.learn(total_timesteps=training_steps)
    model.save(save_path)
    logger.info(f"Custom model trained and saved to: {save_path}")

class EnhancedTimetableDemo:
    """Orchestrates the hybrid AI timetable generation process."""
    def __init__(self, config: Optional[EnhancedConfig] = None):
        self.config = config if config else EnhancedConfig()
        self.constraint_solver = ConstraintSolver(self.config)
        self.nlp_parser = None

    def generate_hybrid_timetable(self, model, env: HybridTimetableEnv, nlp_text: str = ""):
        custom_constraints = self.nlp_parser.parse(nlp_text) if self.nlp_parser and nlp_text else []
        obs, _ = env.reset(options={'custom_constraints': custom_constraints})

        # Generate RL Draft
        lectures_to_place = int(len(env.courses) * self.config.LECTURES_PER_COURSE * self.config.RL_DRAFT_COMPLETION_TARGET)
        for _ in range(lectures_to_place):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc: break
        rl_draft = env.state.copy()

        # Solve with CP-SAT
        final_timetable, stats = self.constraint_solver.solve_with_rl_draft(
            env.courses, env.faculty, env.rooms, env.course_faculty_map, rl_draft, custom_constraints)

        # Calculate final stats
        filled = stats.get('filled_slots', 0)
        required = len(env.courses) * self.config.LECTURES_PER_COURSE
        available = env.days * (env.hours_per_day - 1) * env.num_rooms
        completion = filled / required if required > 0 else 0
        utilization = int(np.count_nonzero(final_timetable)) / available if available > 0 else 0
        final_stats = {"task_completion_rate": completion, "utilization_rate": utilization, "solver_stats": stats}

        return final_timetable, final_stats

# =============================================================================
# STANDALONE EXECUTION (FOR TESTING)
# =============================================================================
def main_test():
    """Function to test the engine independently of the Flask app."""
    logger.info("--- Starting Standalone Timetable Engine Test ---")

    # Define demo data
    courses = ["Intro to AI", "Web Dev", "Robotics", "Signal Proc.", "Cybersecurity"]
    faculty = ["Prof. Turing", "Prof. Berners-Lee", "Prof. Asimov", "Prof. Shannon"]
    rooms = ["Room 101", "Room 202", "Robotics Lab", "Security Ops Center"]
    course_faculty_map = {
        "Intro to AI": "Prof. Turing", "Web Dev": "Prof. Berners-Lee",
        "Robotics": "Prof. Asimov", "Signal Proc.": "Prof. Shannon",
        "Cybersecurity": "Prof. Turing"
    }

    # Test model training
    test_model_path = "instance/test_model.zip"
    train_and_save_model(courses, faculty, rooms, course_faculty_map, save_path=test_model_path)

    # Test generation
    config = EnhancedConfig()
    demo = EnhancedTimetableDemo(config)
    demo.nlp_parser = NLPParser(courses, faculty, rooms)
    env = HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, config)
    model = PPO.load(test_model_path, env=env)

    nlp_request = "Robotics must be in Robotics Lab and Prof. Turing prefers mornings."
    timetable, stats = demo.generate_hybrid_timetable(model, env, nlp_text=nlp_request)

    # Display results
    if timetable is not None:
        print(f"\n--- Generation Stats --- \n{stats}\n")
        # A simple text-based display for testing
        for r_idx, room in enumerate(rooms):
            print(f"\n--- {room} ---")
            for d in range(config.DAYS):
                day_schedule = [env.int_to_course.get(timetable[d, h, r_idx], '---') for h in range(config.HOURS_PER_DAY) if h != 4]
                print(f"Day {d+1}: {' | '.join(s[:7].ljust(7) for s in day_schedule)}")
    else:
        print("Timetable generation failed.")

    logger.info("--- Standalone Test Finished ---")

if __name__ == "__main__":
    main_test()

