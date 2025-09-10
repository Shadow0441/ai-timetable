rom dataclasses import dataclass

@dataclass
class Config:
    # Database
    DB_URL: str = "mysql+pymysql://root:VsalZewdQsSTPqbJIBsNigzXNNaCNpXz@metro.proxy.rlwy.net:58966/railway"

    # Timetable constraints
    DAYS: int = 5
    HOURS_PER_DAY: int = 8
    LECTURES_PER_COURSE: int = 3

    # RL Training - DEMO OPTIMIZED
    MODEL_PATH: str = "demo_timetable_agent.zip"
    BACKUP_MODEL_PATH: str = "pretrained_model.zip"
    LEARNING_RATE: float = 0.001

    # Demo mode settings
    DEMO_MODE: bool = True  # Set to False for intensive training
    DEMO_MAX_TIME: int = 45  # Max 45 seconds for demo

    # Training steps
    INTENSIVE_TRAINING: int = 1000000  # For pre-training (run once at home)
    DEMO_TRAINING: int = 5000  # Quick adaptation for demo

    # Rewards
    VALID_PLACEMENT: float = 100.0
    COMPLETION_BONUS: float = 2000.0
    PROGRESS_REWARD: float = 20.0
    CONFLICT_PENALTY: float = -30.0  # Gentler penalties
    FACULTY_CLASH_PENALTY: float = -40.0

# fast_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from config import Config

class DemoTimetableEnv(gym.Env):
    def __init__(self, courses: List[str], faculty: List[str], rooms: List[str],
                 course_faculty_map: Dict[str, str], config: Config = Config()):
        super().__init__()

        self.config = config
        self.courses = courses
        self.faculty = faculty
        self.rooms = rooms
        self.course_faculty_map = course_faculty_map

        # Environment dimensions
        self.days = config.DAYS
        self.hours_per_day = config.HOURS_PER_DAY
        self.num_rooms = len(rooms)

        # Mappings
        self.course_to_int = {course: i + 1 for i, course in enumerate(courses)}
        self.int_to_course = {i + 1: course for i, course in enumerate(courses)}
        self.faculty_to_int = {fac: i + 1 for i, fac in enumerate(faculty)}

        # Optimized action and observation spaces
        self.action_space = spaces.MultiDiscrete([self.days, self.hours_per_day, self.num_rooms])
        self.observation_space = spaces.Box(
            low=0, high=len(courses) + 1,
            shape=(self.days, self.hours_per_day, self.num_rooms),
            dtype=np.int32
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Initialize state
        self.state = np.zeros((self.days, self.hours_per_day, self.num_rooms), dtype=np.int32)
        self.faculty_schedule = np.zeros((self.days, self.hours_per_day), dtype=np.int32)

        # Create class instances with smart distribution
        self.class_instances = self._create_balanced_classes()
        self.current_class_idx = 0
        self.steps_taken = 0
        self.max_steps = len(self.class_instances) * 8  # Reasonable limit

        return self.state.copy(), {}

    def _create_balanced_classes(self) -> List[str]:
        """Create a balanced, easier-to-schedule distribution"""
        instances = []

        # Create instances with some structure to make learning easier
        for course in self.courses:
            for _ in range(self.config.LECTURES_PER_COURSE):
                instances.append(course)

        # Smart shuffle - keeps some structure while randomizing
        random.shuffle(instances)
        return instances

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        day, hour, room_idx = int(action[0]), int(action[1]), int(action[2])
        reward = 0.0
        terminated = False
        truncated = False

        self.steps_taken += 1

        # Timeout with progress reward
        if self.steps_taken >= self.max_steps:
            truncated = True
            progress_bonus = (self.current_class_idx / len(self.class_instances)) * 1000
            return self.state.copy(), progress_bonus - 50, terminated, truncated, {"reason": "timeout"}

        # Check completion
        if self.current_class_idx >= len(self.class_instances):
            terminated = True
            return self.state.copy(), self.config.COMPLETION_BONUS, terminated, truncated, {"reason": "completed"}

        # Get current class
        course_to_place = self.class_instances[self.current_class_idx]
        course_int = self.course_to_int[course_to_place]
        faculty_name = self.course_faculty_map.get(course_to_place, "Unknown")
        faculty_int = self.faculty_to_int.get(faculty_name, 0)

        # Constraint checking with gentle penalties
        if self.state[day, hour, room_idx] != 0:
            reward = self.config.CONFLICT_PENALTY
            return self.state.copy(), reward, terminated, truncated, {"violation": "slot_occupied"}

        if self.faculty_schedule[day, hour] != 0 and self.faculty_schedule[day, hour] != faculty_int:
            reward = self.config.FACULTY_CLASH_PENALTY
            return self.state.copy(), reward, terminated, truncated, {"violation": "faculty_clash"}

        # SUCCESSFUL PLACEMENT
        self.state[day, hour, room_idx] = course_int
        self.faculty_schedule[day, hour] = faculty_int
        self.current_class_idx += 1

        # Generous rewards for demo success
        base_reward = self.config.VALID_PLACEMENT
        progress_multiplier = 1 + (self.current_class_idx / len(self.class_instances)) * 2
        reward = base_reward * progress_multiplier

        # Completion bonus
        if self.current_class_idx >= len(self.class_instances):
            terminated = True
            reward += self.config.COMPLETION_BONUS

        return self.state.copy(), reward, terminated, truncated, {}
