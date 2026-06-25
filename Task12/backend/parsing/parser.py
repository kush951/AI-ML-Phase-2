
import re

SKILLS = [
    "python",
    "java",
    "javascript",
    "react",
    "docker",
    "aws",
    "tensorflow",
    "pytorch",
    "sql"
]

def parse_text(text):

    text = text.lower()

    found = []

    for skill in SKILLS:
        pattern = r'\\b' + re.escape(skill) + r'\\b'

        if re.search(pattern, text):
            found.append(skill)

    return {
        "skills": list(set(found))
    }
