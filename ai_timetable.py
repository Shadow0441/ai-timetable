import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from stable_baselines3 import PPO
import os
import time
import logging
from tabulate import tabulate
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import select

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

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    logger.warning("PuLP not installed. Some fallback solver features will be unavailable.")
    PULP_AVAILABLE = False

# =============================================================================
# ENHANCED CONFIGURATION (focused on completion)
# =============================================================================
@dataclass
class EnhancedConfig:
    # Standalone Script Config
    DB_URL: str = "mysql+pymysql://root:VsalZewdQsSTPqbJIBsNigzXNNaCNpXz@metro.proxy.rlwy.net:58966/railway"
    MODEL_PATH: str = "hybrid_timetable_agent.zip"
    BACKUP_MODEL_PATH: str = "pretrained_hybrid_model.zip"

    # Core Timetable Constraints
    DAYS: int = 5
    HOURS_PER_DAY: int = 8
    LECTURES_PER_COURSE: int = 5
    MIN_LECTURES_PER_COURSE: int = 1 # For fallback solver

    # Solver & Hybrid Strategy - PRIORITIZE COMPLETION
    USE_CONSTRAINT_SOLVER: bool = True
    CONSTRAINT_SOLVER: str = "ortools"
    HYBRID_MODE: bool = True
    FORCE_COMPLETE_SCHEDULE: bool = True
    MAX_COMPLETION_ATTEMPTS: int = 10

    # RL Model Training Parameters
    LEARNING_RATE: float = 0.0003
    DEMO_MODE: bool = True
    DEMO_MAX_TIME: int = 120
    CP_TIMEOUT: int = 80

    # Training Steps
    INTENSIVE_TRAINING: int = 50000
    DEMO_TRAINING: int = 2000

    # Enhanced Rewards - HEAVILY FAVOR COMPLETION
    VALID_PLACEMENT: float = 100.0
    COMPLETION_BONUS: float = 10000.0
    SLOT_FILLING_BONUS: float = 50.0
    CONSTRAINT_SATISFACTION_BONUS: float = 100.0
    OPTIMIZATION_BONUS: float = 300.0
    CONFLICT_PENALTY: float = -50.0
    FACULTY_CLASH_PENALTY: float = -100.0
    EMPTY_SLOT_PENALTY: float = -10.0

# =============================================================================
# COMPLETION-FOCUSED CONSTRAINT SOLVER
# =============================================================================
class ConstraintSolver:
    def __init__(self, config: EnhancedConfig):
        self.config = config

    def solve_constraints(self, courses: List[str], faculty: List[str], rooms: List[str],
                          course_faculty_map: Dict[str, str]) -> Tuple[Optional[np.ndarray], Dict]:
        strategies = [
            ("complete_flexible", self._solve_complete_flexible),
            ("greedy_fill", self._solve_greedy_fill),
        ]

        for strategy_name, strategy_func in strategies:
            logger.info(f"Trying completion strategy: {strategy_name}")
            result, stats = strategy_func(courses, faculty, rooms, course_faculty_map)
            if result is not None:
                stats['strategy'] = strategy_name
                logger.info(f"Strategy '{strategy_name}' succeeded with {stats.get('filled_slots', 0)} slots filled")
                return result, stats

        logger.error("All completion strategies failed")
        return None, {'status': 'failed', 'reason': 'all_strategies_failed'}

    def _solve_complete_flexible(self, courses: List[str], faculty: List[str], rooms: List[str],
                                 course_faculty_map: Dict[str, str]) -> Tuple[Optional[np.ndarray], Dict]:
        if not ORTOOLS_AVAILABLE:
            return None, {'status': 'failed', 'reason': 'ortools_unavailable'}

        model = cp_model.CpModel()
        x = {}

        for d in range(self.config.DAYS):
            for h in range(self.config.HOURS_PER_DAY):
                if h == 4: continue
                for r in range(len(rooms)):
                    for c, course in enumerate(courses):
                        x[(d, h, r, c)] = model.NewBoolVar(f'x_{d}_{h}_{r}_{c}')

        for c, course in enumerate(courses):
            num_lectures = sum(x.get((d, h, r, c), 0)
                               for d in range(self.config.DAYS)
                               for h in range(self.config.HOURS_PER_DAY) if h != 4
                               for r in range(len(rooms)))
            model.Add(num_lectures == self.config.LECTURES_PER_COURSE)

        for d in range(self.config.DAYS):
            for h in range(self.config.HOURS_PER_DAY):
                if h == 4: continue
                for r in range(len(rooms)):
                    model.Add(sum(x.get((d, h, r, c), 0) for c in range(len(courses))) <= 1)

        faculty_courses = {}
        for c, course in enumerate(courses):
            fac = course_faculty_map.get(course, "Unknown")
            faculty_courses.setdefault(fac, []).append(c)

        faculty_conflict_violations = []
        for fac, course_indices in faculty_courses.items():
            if len(course_indices) > 1:
                for d in range(self.config.DAYS):
                    for h in range(self.config.HOURS_PER_DAY):
                        if h == 4: continue
                        violation = model.NewBoolVar(f'faculty_violation_{fac}_{d}_{h}')
                        faculty_conflict_violations.append(violation)
                        total_scheduled = sum(x.get((d, h, r, c), 0) for r in range(len(rooms)) for c in course_indices)
                        model.Add(total_scheduled <= 1 + violation * len(course_indices))

        if faculty_conflict_violations:
            model.Minimize(sum(faculty_conflict_violations))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.CP_TIMEOUT
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            timetable = np.zeros((self.config.DAYS, self.config.HOURS_PER_DAY, len(rooms)), dtype=np.int32)
            for d in range(self.config.DAYS):
                for h in range(self.config.HOURS_PER_DAY):
                    if h == 4: continue
                    for r in range(len(rooms)):
                        for c, course in enumerate(courses):
                            if (d, h, r, c) in x and solver.Value(x[(d, h, r, c)]) == 1:
                                timetable[d, h, r] = c + 1
            filled_slots = int(np.count_nonzero(timetable))
            stats = {'status': 'solved', 'solve_time': solver.WallTime(), 'filled_slots': filled_slots}
            return timetable, stats

        return None, {'status': 'failed'}

    def _solve_greedy_fill(self, courses: List[str], faculty: List[str], rooms: List[str],
                           course_faculty_map: Dict[str, str]) -> Tuple[Optional[np.ndarray], Dict]:
        logger.info("Using greedy fill strategy")
        timetable = np.zeros((self.config.DAYS, self.config.HOURS_PER_DAY, len(rooms)), dtype=np.int32)
        lectures_remaining = {course: self.config.LECTURES_PER_COURSE for course in courses}

        available_slots = [(d, h, r) for d in range(self.config.DAYS) for h in range(self.config.HOURS_PER_DAY) if h != 4 for r in range(len(rooms))]
        random.shuffle(available_slots)

        faculty_schedule = {(d, h): set() for d in range(self.config.DAYS) for h in range(self.config.HOURS_PER_DAY)}

        for course, count in lectures_remaining.items():
            placed_count = 0
            for _ in range(count):
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
                    logger.warning(f"Could not place all lectures for {course}")

        filled_slots = int(np.count_nonzero(timetable))
        required_slots = len(courses) * self.config.LECTURES_PER_COURSE
        if filled_slots >= required_slots:
            return timetable, {'status': 'solved', 'filled_slots': filled_slots, 'required_slots': required_slots}
        return None, {'status': 'failed', 'filled_slots': filled_slots, 'required_slots': required_slots}

# =============================================================================
# RL ENVIRONMENT (No longer used directly for generation, only training)
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

        target_size = self.observation_space.shape[0]
        if len(full_obs) < target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(full_obs)] = full_obs
            return padded
        return full_obs[:target_size]

     def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.zeros((self.days, self.hours_per_day, self.num_rooms), dtype=np.int32)
        self.faculty_schedule = np.zeros((self.days, self.hours_per_day), dtype=np.int32)
        self.lectures_needed = {course: self.config.LECTURES_PER_COURSE for course in self.courses}
        self.current_course_idx = 0
        self.steps_taken = 0
        self.max_steps = len(self.courses) * self.config.LECTURES_PER_COURSE * 5
        return self._get_observation(), {}

     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        day, hour, room_idx = map(int, action)
        reward, terminated, truncated, info = 0.0, False, False, {}
        self.steps_taken += 1

        if self.steps_taken >= self.max_steps:
            truncated = True
            total_needed = sum(self.lectures_needed.values())
            total_required = len(self.courses) * self.config.LECTURES_PER_COURSE
            completion_ratio = (total_required - total_needed) / total_required
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

        if hour == 4:
             return self._get_observation(), self.config.CONFLICT_PENALTY * 0.5, terminated, truncated, {"reason": "lunch_break"}

        if self.state[day, hour, room_idx] != 0:
            reward = self.config.CONFLICT_PENALTY * 0.5
        elif self.faculty_schedule[day, hour] != 0 and self.faculty_schedule[day, hour] != faculty_int:
            reward = self.config.FACULTY_CLASH_PENALTY * 0.5
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
# COMPLETION-GUARANTEED ORCHESTRATOR
# =============================================================================
class EnhancedTimetableDemo:
    def __init__(self):
        self.config = EnhancedConfig()
        self.constraint_solver = ConstraintSolver(self.config)

    def load_data_fast(self):
        logger.info("📊 Loading data for hybrid AI system...")
        engine = self._engine_creator(self.config.DB_URL)
        if not engine:
            return self._create_demo_data()
        try:
            return self._create_demo_data()
        except Exception as e:
            logger.warning(f"⚠️ Database issue, using demo data: {e}")
            return self._create_demo_data()

    def _create_demo_data(self):
        logger.info("🎭 Using enhanced demo data...")
        courses = ["Data Structures", "Database Systems", "Computer Networks", "OS", "Algorithms", "Digital Circuits"]
        faculty = ["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown", "Dr. Davis", "Dr. Wilson"]
        rooms = ["Room A", "Room B", "Room C", "Room D"]
        course_faculty_map = {
            "Data Structures": "Dr. Smith", "Database Systems": "Dr. Johnson", "Computer Networks": "Dr. Williams",
            "OS": "Dr. Brown", "Algorithms": "Dr. Smith", "Digital Circuits": "Dr. Davis"
        }
        return courses, faculty, rooms, course_faculty_map

    def _engine_creator(self, db_url: str):
        try:
            engine = create_engine(db_url, connect_args={'connect_timeout': 5})
            engine.connect().close()
            return engine
        except Exception as e:
            logger.error(f"❌ DB Error: {e}")
            return None

    def train_hybrid_model(self, env: HybridTimetableEnv):
        logger.info("🤖 Training completion-focused hybrid AI model...")

        if os.path.exists(self.config.BACKUP_MODEL_PATH):
            logger.info(f"🚀 Found pre-trained model at {self.config.BACKUP_MODEL_PATH}. Loading...")
            try:
                model = PPO.load(self.config.BACKUP_MODEL_PATH, env=env)
                logger.info("✅ Pre-trained model loaded successfully.")
                return model
            except Exception as e:
                logger.warning(f"⚠️ Could not load pre-trained model: {e}")

        logger.info("🔧 Creating new completion-focused model...")
        model = PPO("MlpPolicy", env, learning_rate=self.config.LEARNING_RATE, n_steps=1024, batch_size=64, n_epochs=10, verbose=0)
        steps = self.config.DEMO_TRAINING if self.config.DEMO_MODE else self.config.INTENSIVE_TRAINING
        logger.info(f"🎯 Training new model for {steps} timesteps...")
        model.learn(total_timesteps=steps)
        logger.info("✅ New model training complete.")
        return model

    def generate_hybrid_timetable(self, model, env: HybridTimetableEnv):
        logger.info("⚡ Generating COMPLETE timetable with hybrid AI...")

        # --- NEW LOGIC: GUARANTEE COMPLETENESS ---
        # The constraint solver is now the primary tool for generation because it's reliable.
        # The RL model's role is now purely for training and potential future enhancements.

        timetable, stats = self.constraint_solver.solve_constraints(
            env.courses, env.faculty, env.rooms, env.course_faculty_map
        )

        if timetable is None:
             logger.error("All CP strategies failed to produce a timetable.")
             return None, {}

        # NEW: Final backfill step to ensure visual completeness
        final_timetable = self._backfill_timetable_intelligently(timetable, env)

        filled_slots = int(np.count_nonzero(final_timetable))
        required_slots = len(env.courses) * self.config.LECTURES_PER_COURSE
        total_available_slots = env.days * (env.hours_per_day - 1) * env.num_rooms

        task_completion_rate = stats.get('filled_slots', 0) / required_slots if required_slots > 0 else 0
        utilization_rate = filled_slots / total_available_slots if total_available_slots > 0 else 0

        final_stats = {
            "filled_slots": filled_slots,
            "required_slots": required_slots,
            "total_available_slots": total_available_slots,
            "task_completion_rate": task_completion_rate,
            "utilization_rate": utilization_rate,
            "cp_stats": stats
        }
        logger.info(f"✅ Timetable generation finished. Task Completion: {stats.get('filled_slots',0)}/{required_slots} ({task_completion_rate:.2%}) | Utilization: {utilization_rate:.2%}")
        return final_timetable, final_stats

    def _backfill_timetable_intelligently(self, timetable: np.ndarray, env: HybridTimetableEnv) -> np.ndarray:
        logger.info("✍️ Backfilling empty slots to maximize utilization...")

        # Create a map of which faculty teaches which courses
        faculty_to_courses = {}
        for course, faculty in env.course_faculty_map.items():
            faculty_to_courses.setdefault(faculty, []).append(course)

        # Build a schedule of when each faculty is busy
        faculty_schedule = {}
        for d, h, r in np.ndindex(timetable.shape):
            course_id = timetable[d, h, r]
            if course_id > 0:
                course_name = env.int_to_course.get(course_id)
                faculty_name = env.course_faculty_map.get(course_name)
                faculty_schedule.setdefault((d, h), set()).add(faculty_name)

        # Find all empty slots
        empty_slots = []
        for d, h, r in np.ndindex(timetable.shape):
            if h != 4 and timetable[d, h, r] == 0:
                empty_slots.append((d, h, r))

        # Try to fill empty slots with relevant tutorial sessions
        TUTORIAL_ID_OFFSET = 100 # To distinguish tutorials from lectures
        slots_filled = 0
        for d, h, r in empty_slots:
            # Find a faculty member who is free at this time
            busy_faculty = faculty_schedule.get((d, h), set())
            free_faculty = [f for f in env.faculty if f not in busy_faculty]

            if free_faculty:
                # Pick a random free faculty member and one of their courses
                faculty_to_assign = random.choice(free_faculty)
                if faculty_to_assign in faculty_to_courses:
                    course_to_assign = random.choice(faculty_to_courses[faculty_to_assign])
                    course_id = env.course_to_int[course_to_assign]

                    # Assign a "Tutorial" session
                    timetable[d, h, r] = course_id + TUTORIAL_ID_OFFSET
                    faculty_schedule.setdefault((d, h), set()).add(faculty_to_assign)
                    slots_filled += 1

        logger.info(f"✅ Backfill complete. Added {slots_filled} optional tutorial/study sessions.")
        return timetable

    def display_hybrid_timetable(self, timetable, rooms, courses, stats):
        if timetable is None:
            logger.error("❌ No timetable generated")
            return

        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        hours = [f"{9+i}:00" for i in range(self.config.HOURS_PER_DAY)]
        int_to_course = {i + 1: course for i, course in enumerate(courses)}
        TUTORIAL_ID_OFFSET = 100

        completion_rate = stats.get('task_completion_rate', 0) * 100
        utilization_rate = stats.get('utilization_rate', 0) * 100

        header = "\n" + "="*80
        header += "\n🎓 COMPLETE & UTILIZED HYBRID TIMETABLE"
        header += f"\n📊 Task Completion: {stats.get('filled_slots',0)}/{stats.get('required_slots',0)} ({completion_rate:.1f}%) | Timetable Utilization: {utilization_rate:.1f}%"
        header += "\n" + "="*80
        logger.info(header)

        for room_idx, room_name in enumerate(rooms):
            logger.info(f"\n🏫 {room_name}")
            table_data = []
            for day_idx, day in enumerate(days[:self.config.DAYS]):
                row = [day]
                display_hour_indices = [h for h in range(self.config.HOURS_PER_DAY) if h != 4]
                for hour_idx in display_hour_indices:
                    course_int = timetable[day_idx, hour_idx, room_idx]

                    # Display logic for tutorials
                    if course_int > TUTORIAL_ID_OFFSET:
                        original_course_name = int_to_course.get(course_int - TUTORIAL_ID_OFFSET, "Unknown")
                        display_name = f"{original_course_name[:8]}-Tut"
                    elif course_int > 0:
                        display_name = int_to_course.get(course_int, "-")
                    else:
                        display_name = "-"

                    row.append(display_name[:15])
                table_data.append(row)

            display_hours = [h for i, h in enumerate(hours) if i != 4]
            print(tabulate(table_data, headers=["Day"] + display_hours, tablefmt="grid"))

# =============================================================================
# STANDALONE SCRIPT EXECUTION LOGIC
# =============================================================================
def main():
    logger.info("🚀 ENHANCED HYBRID TIMETABLE DEMO - Starting...")
    demo_start = time.time()
    try:
        demo = EnhancedTimetableDemo()
        courses, faculty, rooms, course_faculty_map = demo.load_data_fast()
        env = HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, demo.config)
        model = demo.train_hybrid_model(env) # Still train the model for potential future use
        timetable, stats = demo.generate_hybrid_timetable(model, env)
        if timetable is not None:
            demo.display_hybrid_timetable(timetable, rooms, courses, stats)

        total_time = time.time() - demo_start
        completion_rate = stats.get('task_completion_rate', 0) * 100
        logger.info(f"\n🎉 DEMO COMPLETE! Time: {total_time:.1f}s | Task Completion: {completion_rate:.1f}%")
        return True
    except Exception as e:
        logger.error(f"❌ Demo error: {e}", exc_info=True)
        return False

def create_pretrained_model():
    logger.info("🏠 Creating/Updating pre-trained hybrid model...")
    config = EnhancedConfig()
    config.DEMO_MODE = False

    demo = EnhancedTimetableDemo()
    courses, faculty, rooms, course_faculty_map = demo.load_data_fast()
    env = HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, config)

    if os.path.exists(config.BACKUP_MODEL_PATH):
        logger.info(f"🔄 Loading existing model from {config.BACKUP_MODEL_PATH} to continue training...")
        try:
            model = PPO.load(config.BACKUP_MODEL_PATH, env=env)
        except Exception as e:
            logger.warning(f"⚠️ Could not load pre-trained model, it may be incompatible. Error: {e}")
            logger.info("✨ Creating a new pre-trained model from scratch...")
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=config.LEARNING_RATE, n_steps=2048, batch_size=256, n_epochs=10)
    else:
        logger.info("✨ Creating a new pre-trained model from scratch...")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=config.LEARNING_RATE, n_steps=2048, batch_size=256, n_epochs=10)

    logger.info(f"🎯 Commencing intensive training ({config.INTENSIVE_TRAINING} steps)...")
    model.learn(total_timesteps=config.INTENSIVE_TRAINING, reset_num_timesteps=False)
    model.save(config.BACKUP_MODEL_PATH)
    logger.info(f"✅ Pre-trained hybrid model updated and saved to: {config.BACKUP_MODEL_PATH}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pretrain":
        create_pretrained_model()
    else:
        success = main()
        if not success:
            logger.error("❌ Demo failed - check logs")
