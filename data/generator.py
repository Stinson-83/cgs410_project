"""
data/generator.py — Controlled Recursive Sentence Generator

Generates grammatically correct English sentences with increasing syntactic
recursion depth while preserving a core subject–verb dependency. Multiple
templates ensure diversity.

References:
    - Subject relative clauses: "The dog that chased the cat barked."
    - Object relative clauses: "The man saw the dog that chased the cat."
    - Prepositional phrase attachment: "The boy on the roof near the chimney laughed."
"""

import random
from typing import List, Tuple, Dict

# ─── Vocabulary pools ────────────────────────────────────────────────────────────

ANIMATE_NOUNS = [
    "dog", "cat", "boy", "girl", "man", "woman", "bird", "fox",
    "mouse", "horse", "teacher", "doctor", "chef", "knight", "queen",
    "farmer", "child", "student", "artist", "soldier",
]

INANIMATE_NOUNS = [
    "book", "table", "roof", "car", "fence", "house", "bridge",
    "garden", "tower", "wall", "door", "river", "hill", "stone",
    "window", "tree", "lamp", "bell", "path", "gate",
]

TRANSITIVE_VERBS = [
    "chased", "watched", "followed", "noticed", "admired",
    "praised", "called", "helped", "found", "loved",
]

INTRANSITIVE_VERBS = [
    "barked", "laughed", "smiled", "shouted", "slept",
    "sang", "jumped", "waited", "ran", "danced",
]

PREPOSITIONS = [
    "on", "near", "behind", "beside", "under",
    "above", "inside", "around", "beyond", "beneath",
]


# ─── Template: Subject Relative Clause ───────────────────────────────────────────

def _generate_subject_rc(depth: int, rng: random.Random) -> str:
    """
    Generates a sentence with nested subject relative clauses.

    Depth 1: "The dog barked."
    Depth 2: "The dog that chased the cat barked."
    Depth 3: "The dog that chased the cat that watched the mouse barked."

    The main verb is always at the end, attached to the outermost subject.
    """
    nouns = rng.sample(ANIMATE_NOUNS, depth + 1)
    verbs_trans = rng.sample(TRANSITIVE_VERBS, depth - 1) if depth > 1 else []
    main_verb = rng.choice(INTRANSITIVE_VERBS)

    # Build from innermost to outermost
    # Start with the main subject
    sentence = f"The {nouns[0]}"

    # Add relative clauses
    for i in range(depth - 1):
        sentence += f" that {verbs_trans[i]} the {nouns[i + 1]}"

    sentence += f" {main_verb}."
    return sentence


# ─── Template: Object Relative Clause ────────────────────────────────────────────

def _generate_object_rc(depth: int, rng: random.Random) -> str:
    """
    Generates a sentence with nested object relative clauses.

    Depth 1: "The man saw the dog."
    Depth 2: "The man saw the dog that chased the cat."
    Depth 3: "The man saw the dog that chased the cat that watched the mouse."
    """
    nouns = rng.sample(ANIMATE_NOUNS, depth + 1)
    verbs = rng.sample(TRANSITIVE_VERBS, depth)

    sentence = f"The {nouns[0]} {verbs[0]} the {nouns[1]}"

    for i in range(1, depth - 1):
        sentence += f" that {verbs[i]} the {nouns[i + 1]}"

    # For depth >= 2, the last verb needs an object if we have enough nouns
    if depth >= 2:
        sentence += f" that {verbs[-1]} the {nouns[-1]}" if len(nouns) > 2 else ""

    sentence += "."
    return sentence


# ─── Template: Prepositional Phrase Stacking ─────────────────────────────────────

def _generate_pp_stack(depth: int, rng: random.Random) -> str:
    """
    Generates sentences with stacked prepositional phrases.

    Depth 1: "The boy laughed."
    Depth 2: "The boy on the roof laughed."
    Depth 3: "The boy on the roof near the chimney laughed."
    """
    subject = rng.choice(ANIMATE_NOUNS)
    main_verb = rng.choice(INTRANSITIVE_VERBS)
    preps = rng.sample(PREPOSITIONS, min(depth - 1, len(PREPOSITIONS)))
    pp_nouns = rng.sample(INANIMATE_NOUNS, min(depth - 1, len(INANIMATE_NOUNS)))

    sentence = f"The {subject}"

    for i in range(depth - 1):
        sentence += f" {preps[i]} the {pp_nouns[i]}"

    sentence += f" {main_verb}."
    return sentence


# ─── Template Registry ───────────────────────────────────────────────────────────

TEMPLATES = {
    "subject_rc": _generate_subject_rc,
    "object_rc": _generate_object_rc,
    "pp_stack": _generate_pp_stack,
}


# ─── Public API ──────────────────────────────────────────────────────────────────

def generate_sentences(
    max_depth: int,
    num_per_depth: int,
    seed: int = 42,
) -> Dict[int, List[Tuple[str, str]]]:
    """
    Generate sentences at each recursion depth using multiple templates.

    Args:
        max_depth: Maximum recursion depth (1-indexed).
        num_per_depth: Number of sentences to generate per depth level.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping depth → list of (sentence, template_name) tuples.
    """
    rng = random.Random(seed)
    result: Dict[int, List[Tuple[str, str]]] = {}

    template_names = list(TEMPLATES.keys())

    for depth in range(1, max_depth + 1):
        sentences = []
        for _ in range(num_per_depth):
            # Cycle through templates for diversity
            template_name = rng.choice(template_names)
            gen_fn = TEMPLATES[template_name]
            sentence = gen_fn(depth, rng)
            sentences.append((sentence, template_name))
        result[depth] = sentences

    return result


# ─── CLI test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = generate_sentences(max_depth=5, num_per_depth=3, seed=42)
    for depth, sents in data.items():
        print(f"\n--- Depth {depth} ---")
        for sent, tmpl in sents:
            print(f"  [{tmpl}] {sent}")
