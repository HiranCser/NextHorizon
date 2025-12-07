from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model once
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading embedding model: {model_name}")
st_model = SentenceTransformer(model_name)


def to_list(skill_str: str) -> List[str]:
    """
    Convert comma-separated skills string to list of trimmed skills.
    """
    if not isinstance(skill_str, str):
        return []
    return [s.strip() for s in skill_str.split(",") if s.strip()]


def normalize_skill_text(skill: str) -> str:
    """
    Basic normalization: lowercase and strip. You can extend this.
    """
    return skill.strip().lower()


def compute_embeddings(
    model: SentenceTransformer,
    skills: List[str]
) -> np.ndarray:
    """
    Compute normalized embeddings for a list of skill phrases.
    """
    if not skills:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    norm_skills = [normalize_skill_text(s) for s in skills]
    emb = model.encode(norm_skills, normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)


def calculate_skill_gaps(
    required: List[str],
    candidate: List[str],
    model: SentenceTransformer = st_model,
    partial_threshold: float = 0.55
)-> tuple[List[str], List[str]]:
    """
    For a single row:
    - required: list of required skills
    - candidate: list of candidate skills
    Uses sentence embeddings + cosine similarity.
    Treats skills with max similarity < partial_threshold as gaps.
    Returns the list of required skills considered gaps.
    """
    if not required:
        return [], []

    req_emb = compute_embeddings(model, required)
    cand_emb = compute_embeddings(model, candidate)

    if cand_emb.shape[0] == 0:
        # No candidate skills: all required are gaps
        return list(required), []

    gaps = []
    matches = []

    # For each required skill, find max similarity with candidate skills
    # cosine similarity = dot product (because embeddings are normalized)
    sim_matrix = np.matmul(req_emb, cand_emb.T)  # [num_req, num_cand]

    for i, r in enumerate(required):
        sims = sim_matrix[i]
        max_sim = float(np.max(sims))  # best match
        if max_sim < partial_threshold:
            gaps.append(r)
        else:
            matches.append(r)

    return gaps, matches

