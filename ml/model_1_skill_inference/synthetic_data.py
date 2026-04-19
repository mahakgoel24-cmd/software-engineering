import random

def generate_synthetic_submission():
    templates = [
        "I designed a modular architecture using {tech} focusing on {focus}.",
        "The solution emphasizes {focus} and uses {tech} for scalability.",
        "I implemented a clean design with attention to {focus} using {tech}.",
        "Focused on {focus} while integrating {tech} for robustness."
    ]

    tech_stack = ["FastAPI", "React", "PostgreSQL", "Docker", "Figma"]
    focus_areas = ["clarity", "performance", "maintainability", "UX", "scalability"]

    tech = random.choice(tech_stack)
    focus = random.choice(focus_areas)

    text = random.choice(templates).format(tech=tech, focus=focus)

    # Metadata
    time_spent = round(random.uniform(1.0, 8.0), 2)
    revisions = random.randint(1, 5)
    difficulty = random.randint(1, 3)

    # Synthetic "true" scores (what model learns from)
    clarity = min(1.0, 0.4 + (focus == "clarity") * 0.4 + random.random() * 0.2)
    depth = min(1.0, 0.4 + (tech in ["Docker", "PostgreSQL"]) * 0.4 + random.random() * 0.2)
    creativity = min(1.0, 0.4 + random.random() * 0.6)
    quality = (clarity + depth + creativity) / 3

    return (
        text,
        [time_spent, revisions, difficulty],
        [clarity, depth, creativity, quality]
    )


def generate_dataset(n=500):
    texts, metadata, targets = [], [], []

    for _ in range(n):
        t, m, y = generate_synthetic_submission()
        texts.append(t)
        metadata.append(m)
        targets.append(y)

    return texts, metadata, targets
