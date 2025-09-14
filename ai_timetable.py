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
import pickle

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    logger.warning("OR-Tools not installed. CP-solver features will be unavailable.")
    ORTOOLS_AVAILABLE = False

try:
    from imitation.algorithms import bc
    from imitation.data import types, rollout
    IMITATION_AVAILABLE = True
except ImportError:
    logger.warning("Imitation library not found ('pip install imitation'). Pre-training is unavailable.")
    IMITATION_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class EnhancedConfig:
    # File Paths
    MODEL_PATH: str = "hybrid_timetable_agent.zip"
    BACKUP_MODEL_PATH: str = "pretrained_hybrid_model.zip"
    DATA_FILE_PATH: str = "expert_solutions.pkl"

    # Timetable Structure
    DAYS: int = 5
    HOURS_PER_DAY: int = 8
    LECTURES_PER_COURSE: int = 5

    # Solver Strategy
    CP_TIMEOUT: int = 60

    # RL Training
    LEARNING_RATE: float = 0.0003
    DEMO_MODE: bool = True
    INTENSIVE_TRAINING: int = 50000
    DEMO_TRAINING: int = 2000
    NEW_EXPERT_SOLUTIONS: int = 20
    PRETRAIN_EPOCHS: int = 50

    # Rewards & Penalties
    VALID_PLACEMENT: float = 100.0
    COMPLETION_BONUS: float = 10000.0
    SLOT_FILLING_BONUS: float = 50.0
    OPTIMIZATION_BONUS: float = 300.0
    CONFLICT_PENALTY: float = -50.0
    FACULTY_CLASH_PENALTY: float = -100.0
    EMPTY_SLOT_PENALTY: float = -10.0

# =============================================================================
# CONSTRAINT SOLVER
# =============================================================================
class ConstraintSolver:
    def __init__(self, config: EnhancedConfig):
        self.config = config

    def solve_constraints(self, courses: List[str], faculty: List[str], rooms: List[str],
                          course_faculty_map: Dict[str, str]) -> Tuple[Optional[np.ndarray], Dict]:
        # Attempts to solve using the primary CP-SAT model, falls back to greedy.
        timetable, stats = self._solve_complete_flexible(courses, faculty, rooms, course_faculty_map)
        if timetable is not None:
            return timetable, stats

        logger.warning("CP-SAT solver failed, attempting greedy fallback...")
        return self._solve_greedy_fill(courses, faculty, rooms, course_faculty_map)

    def _solve_complete_flexible(self, courses: List[str], faculty: List[str], rooms: List[str],
                                   course_faculty_map: Dict[str, str]) -> Tuple[Optional[np.ndarray], Dict]:
        if not ORTOOLS_AVAILABLE: return None, {}

        model = cp_model.CpModel()
        x = {} # Decision variables

        for d in range(self.config.DAYS):
            for h in range(self.config.HOURS_PER_DAY):
                if h == 4: continue
                for r in range(len(rooms)):
                    for c in range(len(courses)):
                        x[(d, h, r, c)] = model.NewBoolVar(f'x_{d}_{h}_{r}_{c}')

        for c in range(len(courses)):
            model.Add(sum(x.get((d, h, r, c), 0) for d in range(self.config.DAYS) for h in range(self.config.HOURS_PER_DAY) if h != 4 for r in range(len(rooms))) == self.config.LECTURES_PER_COURSE)

        for d in range(self.config.DAYS):
            for h in range(self.config.HOURS_PER_DAY):
                if h == 4: continue
                for r in range(len(rooms)):
                    model.Add(sum(x.get((d, h, r, c), 0) for c in range(len(courses))) <= 1)

        faculty_courses = {fac: [] for fac in faculty}
        for c, course in enumerate(courses):
            faculty_courses.setdefault(course_faculty_map.get(course, "Unknown"), []).append(c)

        violations = []
        for fac, c_indices in faculty_courses.items():
            if len(c_indices) > 1:
                for d in range(self.config.DAYS):
                    for h in range(self.config.HOURS_PER_DAY):
                        if h == 4: continue
                        v = model.NewBoolVar(f'v_{fac}_{d}_{h}')
                        violations.append(v)
                        total_at_time = sum(x.get((d, h, r, c), 0) for r in range(len(rooms)) for c in c_indices)
                        model.Add(total_at_time <= 1 + v * len(c_indices))

        if violations:
            model.Minimize(sum(violations))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.CP_TIMEOUT
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            timetable = np.zeros((self.config.DAYS, self.config.HOURS_PER_DAY, len(rooms)), dtype=np.int32)
            for (d, h, r, c), var in x.items():
                if solver.Value(var) == 1:
                    timetable[d, h, r] = c + 1
            stats = {'status': 'solved', 'solve_time': solver.WallTime(), 'filled_slots': int(np.count_nonzero(timetable))}
            return timetable, stats

        return None, {'status': 'failed'}

    def _solve_greedy_fill(self, courses: List[str], faculty: List[str], rooms: List[str],
                           course_faculty_map: Dict[str, str]) -> Tuple[Optional[np.ndarray], Dict]:
        timetable = np.zeros((self.config.DAYS, self.config.HOURS_PER_DAY, len(rooms)), dtype=np.int32)
        lectures = {c: self.config.LECTURES_PER_COURSE for c in courses}
        slots = [(d, h, r) for d in range(self.config.DAYS) for h in range(self.config.HOURS_PER_DAY) if h != 4 for r in range(len(rooms))]
        random.shuffle(slots)
        fac_schedule = {}

        for course in courses:
            for _ in range(lectures[course]):
                for i, (d, h, r) in enumerate(slots):
                    if timetable[d, h, r] == 0:
                        fac_name = course_faculty_map.get(course)
                        if fac_name not in fac_schedule.get((d, h), set()):
                            timetable[d, h, r] = courses.index(course) + 1
                            fac_schedule.setdefault((d, h), set()).add(fac_name)
                            slots.pop(i)
                            break

        filled = int(np.count_nonzero(timetable))
        return timetable, {'status': 'greedy_solved', 'filled_slots': filled}

# =============================================================================
# RL ENVIRONMENT
# =============================================================================
class HybridTimetableEnv(gym.Env):
    # This class defines the simulation environment for the RL agent. It is not changed.
    def __init__(self, courses: List[str], faculty: List[str], rooms: List[str],
                 course_faculty_map: Dict[str, str], config: EnhancedConfig):
        super().__init__()
        self.config, self.courses, self.faculty, self.rooms, self.course_faculty_map = config, courses, faculty, rooms, course_faculty_map
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
        course_info, total_needed = [], sum(self.lectures_needed.values())
        total_required = len(self.courses) * self.config.LECTURES_PER_COURSE
        total_placed = total_required - total_needed
        for course in self.courses:
            needed = self.lectures_needed.get(course, 0)
            course_info.extend([needed, self.config.LECTURES_PER_COURSE - needed])
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
        self.lectures_needed = {c: self.config.LECTURES_PER_COURSE for c in self.courses}
        self.current_course_idx, self.steps_taken = 0, 0
        self.max_steps = len(self.courses) * self.config.LECTURES_PER_COURSE * 5
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        day, hour, room_idx = map(int, action)
        reward, terminated, truncated, info = 0.0, False, False, {}
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            truncated = True
            needed = sum(self.lectures_needed.values())
            required = len(self.courses) * self.config.LECTURES_PER_COURSE
            ratio = (required - needed) / required if required > 0 else 0
            reward = ratio * self.config.COMPLETION_BONUS + self.config.EMPTY_SLOT_PENALTY * needed
            return self._get_observation(), reward, terminated, truncated, {"reason": "timeout"}
        available = [c for c, n in self.lectures_needed.items() if n > 0]
        if not available:
            return self._get_observation(), self.config.COMPLETION_BONUS, True, truncated, {"reason": "completed"}
        course = available[self.current_course_idx % len(available)]
        c_int = self.course_to_int[course]
        f_name = self.course_faculty_map.get(course, "Unknown")
        f_int = self.faculty_to_int.get(f_name, 0)
        if hour == 4: return self._get_observation(), self.config.CONFLICT_PENALTY * 0.5, False, truncated, {"reason": "lunch"}
        if self.state[day, hour, room_idx] != 0: reward = self.config.CONFLICT_PENALTY
        elif self.faculty_schedule[day, hour] != 0 and self.faculty_schedule[day, hour] != f_int: reward = self.config.FACULTY_CLASH_PENALTY
        else:
            self.state[day, hour, room_idx] = c_int
            self.faculty_schedule[day, hour] = f_int
            self.lectures_needed[course] -= 1
            self.current_course_idx += 1
            reward = self.config.VALID_PLACEMENT + self.config.SLOT_FILLING_BONUS
            if self.lectures_needed[course] == 0: reward += self.config.OPTIMIZATION_BONUS
        return self._get_observation(), reward, terminated, truncated, info

# =============================================================================
# ORCHESTRATOR & DEMO
# =============================================================================
class EnhancedTimetableDemo:
    def __init__(self):
        self.config = EnhancedConfig()
        self.constraint_solver = ConstraintSolver(self.config)

    def load_data_fast(self):
        return self._create_demo_data()

    def _create_demo_data(self):
        # MODIFIED: Room names are now more descriptive to support intelligent backfilling.
        logger.info("🎭 Using enhanced demo data with specific lab rooms...")
        courses = ["Data Structures", "Database Systems", "Computer Networks", "OS", "Algorithms", "Digital Circuits"]
        faculty = ["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown", "Dr. Davis", "Dr. Wilson"]
        rooms = ["Lecture Hall 101", "Room 102", "Computer Lab A", "Electronics Lab B"]
        course_faculty_map = {
            "Data Structures": "Dr. Smith", "Database Systems": "Dr. Johnson", "Computer Networks": "Dr. Williams",
            "OS": "Dr. Brown", "Algorithms": "Dr. Smith", "Digital Circuits": "Dr. Davis"
        }
        return courses, faculty, rooms, course_faculty_map

    def train_hybrid_model(self, env: HybridTimetableEnv):
        logger.info(f"🤖 Loading best available model from {self.config.BACKUP_MODEL_PATH}...")
        if not os.path.exists(self.config.BACKUP_MODEL_PATH):
            logger.error("❌ Model not found. Please run with the 'pretrain' argument first.")
            return None
        try:
            return PPO.load(self.config.BACKUP_MODEL_PATH, env=env)
        except Exception as e:
            logger.error(f"❌ Could not load model: {e}")
            return None

    def generate_hybrid_timetable(self, model, env: HybridTimetableEnv):
        timetable, stats = self.constraint_solver.solve_constraints(env.courses, env.faculty, env.rooms, env.course_faculty_map)
        if timetable is None:
            logger.error("All solving strategies failed to produce a timetable.")
            return None, {}
        final_timetable = self._backfill_timetable_intelligently(timetable, env)
        required = len(env.courses) * self.config.LECTURES_PER_COURSE
        available = env.days * (env.hours_per_day - 1) * env.num_rooms
        completion = stats.get('filled_slots', 0) / required if required > 0 else 0
        utilization = int(np.count_nonzero(final_timetable)) / available if available > 0 else 0
        final_stats = {"task_completion_rate": completion, "utilization_rate": utilization, "cp_stats": stats}
        logger.info(f"✅ Timetable generation finished. Completion: {completion:.2%} | Utilization: {utilization:.2%}")
        return final_timetable, final_stats

    def _backfill_timetable_intelligently(self, timetable: np.ndarray, env: HybridTimetableEnv) -> np.ndarray:
        # MODIFIED: Schedules 2-hour labs and 1-hour tutorials in two separate passes.
        logger.info("✍️ Backfilling empty slots with optional labs (2h) and tutorials (1h)...")

        faculty_to_courses = {fac: [] for fac in env.faculty}
        for course, faculty in env.course_faculty_map.items():
            faculty_to_courses.setdefault(faculty, []).append(course)
        lab_courses = {"Data Structures", "Computer Networks", "OS", "Digital Circuits"}

        fac_schedule = {}
        for d, h, r in np.ndindex(timetable.shape):
            c_id = timetable[d, h, r]
            if 0 < c_id <= len(env.courses):
                c_name = env.int_to_course.get(c_id)
                f_name = env.course_faculty_map.get(c_name)
                if f_name: fac_schedule.setdefault((d, h), set()).add(f_name)

        TUTORIAL_ID_OFFSET, LAB_ID_OFFSET, LAB_CONTINUATION_ID = 100, 200, -1

        # Pass 1: Schedule 2-Hour Labs
        lab_slots_filled = 0
        possible_lab_starts = []
        for d in range(env.days):
            for h in range(env.hours_per_day - 1):
                if h != 3 and h != 4 and (h + 1) != 4:
                    for r, room_name in enumerate(env.rooms):
                        if "Lab" in room_name and timetable[d, h, r] == 0 and timetable[d, h + 1, r] == 0:
                            possible_lab_starts.append((d, h, r))
        random.shuffle(possible_lab_starts)

        for d, h, r in possible_lab_starts:
            if timetable[d, h, r] == 0 and timetable[d, h + 1, r] == 0:
                busy_now = fac_schedule.get((d, h), set())
                busy_next = fac_schedule.get((d, h + 1), set())
                free_faculty = [f for f in env.faculty if f not in busy_now and f not in busy_next]
                if free_faculty:
                    fac_assign = random.choice(free_faculty)
                    possible_labs = [c for c in faculty_to_courses.get(fac_assign, []) if c in lab_courses]
                    if possible_labs:
                        course_for_lab = random.choice(possible_labs)
                        c_id = env.course_to_int[course_for_lab]
                        timetable[d, h, r] = c_id + LAB_ID_OFFSET
                        timetable[d, h + 1, r] = LAB_CONTINUATION_ID
                        fac_schedule.setdefault((d, h), set()).add(fac_assign)
                        fac_schedule.setdefault((d, h + 1), set()).add(fac_assign)
                        lab_slots_filled += 1

        # Pass 2: Schedule 1-Hour Tutorials
        tutorial_slots_filled = 0
        empty_slots = [(d, h, r) for d, h, r in np.ndindex(timetable.shape) if h != 4 and timetable[d, h, r] == 0]
        random.shuffle(empty_slots)

        for d, h, r in empty_slots:
            if "Lab" in env.rooms[r]: continue
            busy_fac = fac_schedule.get((d, h), set())
            free_fac = [f for f in env.faculty if f not in busy_fac]
            if free_fac:
                fac_assign = random.choice(free_fac)
                courses_can_teach = faculty_to_courses.get(fac_assign, [])
                if courses_can_teach:
                    course_for_tut = random.choice(courses_can_teach)
                    c_id = env.course_to_int[course_for_tut]
                    timetable[d, h, r] = c_id + TUTORIAL_ID_OFFSET
                    fac_schedule.setdefault((d, h), set()).add(fac_assign)
                    tutorial_slots_filled += 1

        logger.info(f"✅ Backfill complete. Added {lab_slots_filled} labs and {tutorial_slots_filled} tutorials.")
        return timetable

    def display_hybrid_timetable(self, timetable, rooms, courses, stats):
        # MODIFIED: Handles display of 2-hour lab blocks.
        if timetable is None: return
        days, hours = ["Mon", "Tue", "Wed", "Thu", "Fri"], [f"{9+i}:00" for i in range(self.config.HOURS_PER_DAY)]
        int_to_course = {i + 1: c for i, c in enumerate(courses)}
        TUTORIAL_ID_OFFSET, LAB_ID_OFFSET, LAB_CONTINUATION_ID = 100, 200, -1

        completion, util = stats.get('task_completion_rate', 0)*100, stats.get('utilization_rate', 0)*100
        header = f"\n{'='*80}\n🎓 HYBRID TIMETABLE\n📊 Completion: {completion:.1f}% | Utilization: {util:.1f}%\n{'='*80}"
        logger.info(header)

        for r_idx, r_name in enumerate(rooms):
            logger.info(f"\n🏛️ {r_name}")
            table = []
            display_hours = [h for i, h in enumerate(hours) if i != 4]
            for d_idx, day in enumerate(days[:self.config.DAYS]):
                row = [day]
                h_idx = 0
                while h_idx < len(display_hours):
                    hour_val = int(display_hours[h_idx].split(':')[0]) - 9 # Convert "10:00" back to index 1
                    c_int = timetable[d_idx, hour_val, r_idx]

                    name = "-"
                    if c_int >= LAB_ID_OFFSET:
                        name = f"{int_to_course.get(c_int - LAB_ID_OFFSET, '?')[:8]}-Lab"
                        row.append(name[:15])
                        row.append("->") # Add continuation marker
                        h_idx += 2 # Important: Skip next hour in display
                        continue
                    elif c_int >= TUTORIAL_ID_OFFSET:
                        name = f"{int_to_course.get(c_int - TUTORIAL_ID_OFFSET, '?')[:8]}-Tut"
                    elif c_int > 0:
                        name = int_to_course.get(c_int, "-")

                    row.append(name[:15])
                    h_idx += 1
                table.append(row)

            print(tabulate(table, headers=["Day"] + display_hours, tablefmt="grid"))

# =============================================================================
# DATA MANAGEMENT & TRAINING PIPELINE
# =============================================================================
def update_and_load_expert_data(config: EnhancedConfig, courses, faculty, rooms, course_faculty_map) -> List[np.ndarray]:
    # Loads expert data, generates new solutions, appends, and saves back.
    if os.path.exists(config.DATA_FILE_PATH):
        with open(config.DATA_FILE_PATH, 'rb') as f:
            expert_solutions = pickle.load(f)
    else:
        expert_solutions = []
    logger.info(f"📚 Loaded {len(expert_solutions)} existing solutions from '{config.DATA_FILE_PATH}'.")
    solver = ConstraintSolver(config)
    for i in range(config.NEW_EXPERT_SOLUTIONS):
        logger.info(f"🧠 Generating new expert solution {i+1}/{config.NEW_EXPERT_SOLUTIONS}...")
        timetable, _ = solver.solve_constraints(courses, faculty, rooms, course_faculty_map)
        if timetable is not None: expert_solutions.append(timetable)
    with open(config.DATA_FILE_PATH, 'wb') as f:
        pickle.dump(expert_solutions, f)
    logger.info(f"💾 Saved! Total dataset size is now {len(expert_solutions)} solutions.")
    return expert_solutions

def convert_solutions_to_trajectories(solutions: List[np.ndarray], env: gym.Env) -> List[types.Trajectory]:
    # Converts solved timetables into the (observation, action) format for imitation learning.
    trajectories = []
    for timetable in solutions:
        obs_list, act_list = [], []
        temp_table = timetable.copy()
        obs, _ = env.reset()
        courses_to_place = [c for c in env.courses for _ in range(env.config.LECTURES_PER_COURSE)]
        for course_name in courses_to_place:
            c_int = env.course_to_int[course_name]
            coords = np.argwhere(temp_table == c_int)
            if len(coords) > 0:
                d, h, r = coords[0]
                obs_list.append(obs)
                act_list.append(np.array([d, h, r]))
                obs, _, _, _, _ = env.step(np.array([d, h, r]))
                temp_table[d, h, r] = -1
        if obs_list:
            trajectories.append(types.Trajectory(obs=np.array(obs_list), acts=np.array(act_list), infos=None, terminal=True))
    return trajectories

def create_pretrained_model():
    """Main iterative training pipeline to create/update the persistent model."""
    logger.info("🏠 Starting iterative pre-training process...")
    config = EnhancedConfig()
    config.DEMO_MODE = False
    if not IMITATION_AVAILABLE:
        logger.error("❌ Imitation library is required. Exiting.")
        return

    demo = EnhancedTimetableDemo()
    courses, faculty, rooms, course_faculty_map = demo.load_data_fast()

    # 1. Update and Load Expert Dataset
    all_solutions = update_and_load_expert_data(config, courses, faculty, rooms, course_faculty_map)
    if not all_solutions:
        logger.error("❌ No expert solutions available to train on. Exiting.")
        return

    # 2. Convert to Trajectories
    logger.info("📜 Converting solutions to trajectories...")
    conv_env = HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, config)
    trajectories = convert_solutions_to_trajectories(all_solutions, conv_env)
    if not trajectories:
        logger.error("❌ Failed to convert solutions to valid trajectories. Exiting.")
        return
    transitions = rollout.flatten_trajectories(trajectories)

    # 3. Pre-train with Behavioral Cloning
    logger.info("🧠 Pre-training model with Behavioral Cloning...")
    venv = DummyVecEnv([lambda: HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, config)])
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        policy_class="MlpPolicy"
    )
    bc_trainer.train(n_epochs=config.PRETRAIN_EPOCHS)

    # 4. Fine-Tune with PPO
    logger.info("🚀 Fine-tuning pre-trained model with PPO...")
    try:
        model = PPO.load(config.BACKUP_MODEL_PATH, env=venv)
        logger.info("🔄 Continuing training from existing model.")
    except:
        logger.info("✨ Creating a new model for fine-tuning.")
        model = PPO("MlpPolicy", venv, verbose=0, learning_rate=config.LEARNING_RATE)

    model.set_policy(bc_trainer.policy) # Load the newly pre-trained policy
    model.learn(total_timesteps=config.INTENSIVE_TRAINING, reset_num_timesteps=False)

    # 5. Save Final Model
    model.save(config.BACKUP_MODEL_PATH)
    logger.info(f"✅ Pre-trained model updated and saved to: {config.BACKUP_MODEL_PATH}")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
def main():
    logger.info("🚀 HYBRID TIMETABLE DEMO - Starting...")
    start = time.time()
    try:
        demo = EnhancedTimetableDemo()
        courses, faculty, rooms, course_faculty_map = demo.load_data_fast()
        env = HybridTimetableEnv(courses, faculty, rooms, course_faculty_map, demo.config)
        model = demo.train_hybrid_model(env)
        if model is None: return False

        timetable, stats = demo.generate_hybrid_timetable(model, env)
        if timetable is not None:
            demo.display_hybrid_timetable(timetable, rooms, courses, stats)

        logger.info(f"\n🎉 DEMO COMPLETE! Time: {time.time() - start:.1f}s")
        return True
    except Exception as e:
        logger.error(f"❌ Demo error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pretrain":
        create_pretrained_model()
    else:
        if not main():
            logger.error("❌ Demo failed - check logs")
