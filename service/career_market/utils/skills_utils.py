from __future__ import annotations

from service.career_market.utils.config import SKILLS_PATH


def _load_skills() -> list[str]:
    if SKILLS_PATH.exists():
        lines = SKILLS_PATH.read_text(encoding="utf-8").splitlines()
        skills = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if skills:
            return skills
    return [
        "Python",
        "SQL",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "TensorFlow",
        "PyTorch",
        "Docker",
        "Kubernetes",
        "AWS",
        "FastAPI",
        "Django",
        "Flask",
        "Git",
        "Linux",
        "React",
        "Node.js",
        "Java",
        "C++",
    ]


SKILLS = _load_skills()
