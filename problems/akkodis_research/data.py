
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


class DataLoader:
    """
    Data access layer for the R&D project staffing optimization model.

    This class reads the required CSV inputs, validates and preprocesses them, and
    exposes clean, typed accessors for the model's sets and parameters:

    Sets
    - I: Employees (list of unique employee identifiers, e.g., "Nom Prenom")
    - J: Projects (list of project names)
    - S: Skills (list of skill names, consistent across both CSV files)

    Parameters
    - a[i, s]: Employee i’s proficiency rating in skill s (0..5), shape (|I|, |S|)
    - R[j, s]: Required headcount for project j who possess skill s (≥0), shape (|J|, |S|)

    Derived Data
    - b[i, s]: Binary possession matrix where b[i, s] = 1 if a[i, s] ≥ 1 else 0,
               shape (|I|, |S|)

    Notes
    - The skills ordering S is taken from the project.csv columns (excluding "Project").
      The employee ratings are re-aligned to that ordering to ensure consistent indexing.
    - Employees are identified by "Nom Prenom" concatenation and must be unique.
    - The class performs strict validation:
      * Skills must match exactly between ip.csv and project.csv.
      * Ratings must be integers in [0, 5].
      * Requirements must be non-negative integers.

    CSV files expected in the working directory:
    - project.csv
    - ip.csv
    """

    # Public attributes (typed) for integrations that prefer direct access
    employees: List[str]
    projects: List[str]
    skills: List[str]

    # Parameter matrices aligned to (I, S) and (J, S) in the defined order
    a_matrix: np.ndarray  # shape (|I|, |S|), dtype=int64
    b_matrix: np.ndarray  # shape (|I|, |S|), dtype=int8
    R_matrix: np.ndarray  # shape (|J|, |S|), dtype=int64

    # Index maps: label -> position
    employee_index: Dict[str, int]
    project_index: Dict[str, int]
    skill_index: Dict[str, int]

    def __init__(self) -> None:
        """
        Initialize the DataLoader by reading and preprocessing CSV inputs.

        Steps performed:
        1. Read project.csv to obtain project skill requirements.
        2. Read ip.csv to obtain employee skill ratings.
        3. Construct canonical skills list S from project.csv and validate that ip.csv
           contains exactly the same skills (order is harmonized to S).
        4. Build employee identifiers as "Nom Prenom" and validate uniqueness.
        5. Validate data types and ranges:
           - Ratings a[i, s] are integers in [0, 5].
           - Requirements R[j, s] are integers in [0, +∞).
        6. Create aligned matrices a (ratings), b (binary possession), and R (requirements).
        7. Create index maps for employees, projects, and skills.

        Raises
        - ValueError: if any validation fails (missing columns, mismatched skills, invalid values).
        """
        # Load raw CSVs
        projects_df = pd.read_csv("project.csv")
        employees_df = pd.read_csv("ip.csv")

        # Basic structural checks
        if "Project" not in projects_df.columns:
            raise ValueError('project.csv must contain a "Project" column.')
        if "Nom" not in employees_df.columns or "Prenom" not in employees_df.columns:
            raise ValueError('ip.csv must contain "Nom" and "Prenom" columns.')

        # Canonical skills from project.csv (preserve CSV column order)
        project_skill_cols = [c for c in projects_df.columns if c != "Project"]
        if len(project_skill_cols) == 0:
            raise ValueError("project.csv contains no skill columns.")

        # Skill columns in ip.csv (exclude identification columns)
        employee_skill_cols = [c for c in employees_df.columns if c not in ("Nom", "Prenom")]
        if len(employee_skill_cols) == 0:
            raise ValueError("ip.csv contains no skill rating columns.")

        # Validate skills set matches exactly
        project_skills_set = set(project_skill_cols)
        employee_skills_set = set(employee_skill_cols)
        if project_skills_set != employee_skills_set:
            missing_in_ip = project_skills_set - employee_skills_set
            missing_in_proj = employee_skills_set - project_skills_set
            raise ValueError(
                "Skill columns mismatch between project.csv and ip.csv.\n"
                f"Missing in ip.csv: {sorted(missing_in_ip)}\n"
                f"Missing in project.csv: {sorted(missing_in_proj)}"
            )

        # Harmonize order of skills to project.csv's ordering
        self.skills = project_skill_cols
        # Construct skill index map
        self.skill_index = {s: k for k, s in enumerate(self.skills)}

        # Normalize identifiers (strip whitespace)
        projects_df["Project"] = projects_df["Project"].astype(str).str.strip()
        employees_df["Nom"] = employees_df["Nom"].astype(str).str.strip()
        employees_df["Prenom"] = employees_df["Prenom"].astype(str).str.strip()

        # Build employee unique identifier
        employees_df["Employee"] = (employees_df["Nom"] + " " + employees_df["Prenom"]).str.strip()

        # Validate employee uniqueness
        if employees_df["Employee"].duplicated().any():
            dups = employees_df[employees_df["Employee"].duplicated()]["Employee"].tolist()
            raise ValueError(f"Duplicate employees found in ip.csv: {dups}")

        # Validate numeric types and ranges for project requirements
        # Convert requirements to integers and ensure non-negativity
        try:
            req_df = projects_df[self.skills].apply(pd.to_numeric, errors="raise")
        except Exception as e:
            raise ValueError(f"Non-numeric values detected in project requirements: {e}")
        if (req_df.isna().any().any()):
            raise ValueError("NaN values detected in project requirements.")
        if (req_df < 0).any().any():
            raise ValueError("Negative values detected in project requirements.")
        # Requirement matrix (J, S)
        self.R_matrix = req_df.to_numpy(dtype=np.int64)

        # Validate numeric types and ranges for employee ratings
        try:
            rate_df = employees_df[self.skills].apply(pd.to_numeric, errors="raise")
        except Exception as e:
            raise ValueError(f"Non-numeric values detected in employee ratings: {e}")
        if (rate_df.isna().any().any()):
            raise ValueError("NaN values detected in employee ratings.")
        if (rate_df < 0).any().any() or (rate_df > 5).any().any():
            raise ValueError("Employee ratings must be integers in [0, 5].")
        # Rating matrix (I, S)
        self.a_matrix = rate_df[self.skills].to_numpy(dtype=np.int64)

        # Binary possession matrix b[i, s] = 1 if a[i, s] >= 1 else 0
        self.b_matrix = (self.a_matrix >= 1).astype(np.int8)

        # Finalize sets
        self.projects = projects_df["Project"].tolist()
        self.employees = employees_df["Employee"].tolist()

        # Index maps
        self.project_index = {p: j for j, p in enumerate(self.projects)}
        self.employee_index = {e: i for i, e in enumerate(self.employees)}

        # Store original DataFrames for potential reference (kept private)
        self._projects_df = projects_df.set_index("Project")
        self._employees_df = employees_df.set_index("Employee")

    # ------------------------------
    # Public API methods (typed)
    # ------------------------------

    def get_employees(self) -> List[str]:
        """
        Return the ordered list of employee identifiers (set I).

        Returns
        - List[str]: Employee names in the canonical order used by matrices.
        """
        return self.employees

    def get_projects(self) -> List[str]:
        """
        Return the ordered list of project identifiers (set J).

        Returns
        - List[str]: Project names in the canonical order used by matrices.
        """
        return self.projects

    def get_skills(self) -> List[str]:
        """
        Return the ordered list of skill names (set S).

        Returns
        - List[str]: Skill names in the canonical order aligned across a, b, and R.
        """
        return self.skills

    def get_rating_matrix(self) -> np.ndarray:
        """
        Get the employee skill rating matrix a[i, s].

        Shape and dtype
        - Shape: (|I|, |S|)
        - Dtype: int64
        - Values: integers in [0, 5]

        Returns
        - np.ndarray: Ratings matrix a.
        """
        return self.a_matrix

    def get_possession_matrix(self) -> np.ndarray:
        """
        Get the binary possession matrix b[i, s], where b[i, s] = 1 if a[i, s] ≥ 1 else 0.

        Shape and dtype
        - Shape: (|I|, |S|)
        - Dtype: int8
        - Values: 0 or 1

        Returns
        - np.ndarray: Binary possession matrix b.
        """
        return self.b_matrix

    def get_requirement_matrix(self) -> np.ndarray:
        """
        Get the project skill requirement matrix R[j, s].

        Shape and dtype
        - Shape: (|J|, |S|)
        - Dtype: int64
        - Values: integers ≥ 0

        Returns
        - np.ndarray: Requirements matrix R.
        """
        return self.R_matrix

    def get_index_maps(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """
        Get dictionaries mapping labels to matrix indices for employees, projects, and skills.

        Returns
        - Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
          (employee_index, project_index, skill_index)
        """
        return self.employee_index, self.project_index, self.skill_index

    def get_employee_dataframe(self) -> pd.DataFrame:
        """
        Return the employee dataframe indexed by employee identifier with skill columns.

        The dataframe includes:
        - Index: "Employee" ("Nom Prenom")
        - Columns: skills S (ratings as integers in [0, 5]), plus "Nom", "Prenom"

        Returns
        - pd.DataFrame: Employee data aligned to the skills ordering.
        """
        # Reorder columns to show "Nom", "Prenom" and skills S
        cols = ["Nom", "Prenom"] + self.skills
        return self._employees_df[cols].copy()

    def get_project_dataframe(self) -> pd.DataFrame:
        """
        Return the project dataframe indexed by project name with requirement columns.

        The dataframe includes:
        - Index: "Project"
        - Columns: skills S (requirements as integers ≥ 0)

        Returns
        - pd.DataFrame: Project requirements aligned to the skills ordering.
        """
        return self._projects_df[self.skills].copy()
