from dataclasses import dataclass

@dataclass
class Config:
    # --- Timetable Structure (Defaults that can be overridden by institute settings) ---
    DAYS: int = 5
    HOURS_PER_DAY: int = 8
    DEFAULT_LECTURES_PER_COURSE: int = 4

    # --- Solver Strategy ---
    CP_TIMEOUT: int = 30  # Timeout for the CP-SAT solver in seconds

