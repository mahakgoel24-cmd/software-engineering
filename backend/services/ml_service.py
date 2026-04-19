from typing import Dict, List
import torch

from backend.database.supabase import supabase

# -------- Model 1 --------
from ml.model_1_skill_inference.inference import infer as model1_infer
from ml.model_1_skill_inference.model import SkillInferenceModel
from ml.common.embeddings import TextEmbedder

# -------- Model 2 --------
from ml.model_2_fair_ranking.inference import rank_candidates

# -------- Model 3 --------
from ml.model_3_qml_optimizer.inference import quantum_select_top_10_percent


class MLService:
    def __init__(self):
        self.embedder = TextEmbedder()

        # MUST MATCH TRAINING
        self.model1 = SkillInferenceModel(
            embedding_dim=384,
            meta_dim=3,
            output_dim=4
        )

        self.model1.load_state_dict(
            torch.load(
                "ml/artifacts/model_1/skill_inference.pt",
                map_location="cpu"
            )
        )
        self.model1.eval()

    def run_pipeline(self, project_id: str) -> Dict:
        # -------------------------
        # Fetch project
        # -------------------------
        project = (
            supabase.table("projects")
            .select("description")
            .eq("id", project_id)
            .single()
            .execute()
        )

        project_description = project.data["description"]

        # -------------------------
        # Fetch submissions
        # -------------------------
        subs = (
            supabase.table("submissions")
            .select("*")
            .eq("project_id", project_id)
            .execute()
        )

        submissions = subs.data
        if not submissions:
            return {"shortlisted": [], "ranking_snapshot": []}

        # -------------------------
        # Fetch developers
        # -------------------------
        developer_ids = [s["developer_id"] for s in submissions]

        devs = (
            supabase.table("developers")
            .select("id, rating, experience")
            .in_("id", developer_ids)
            .execute()
        )

        developers = {d["id"]: d for d in devs.data}

        # -------------------------
        # MODEL 1 — Skill Inference
        # -------------------------
        texts = [s["description"] for s in submissions]
        embeddings = self.embedder.encode(texts)
        project_emb = self.embedder.encode([project_description])[0]

        model1_outputs = []
        semantic_scores = []

        for i in range(len(submissions)):
            out = model1_infer(
                model=self.model1,
                emb=torch.tensor(embeddings[i:i+1], dtype=torch.float32),
                meta=torch.zeros((1, 3), dtype=torch.float32)
            )

            model1_outputs.append(out)

            semantic = torch.cosine_similarity(
                torch.tensor(embeddings[i]),
                torch.tensor(project_emb),
                dim=0
            ).item()

            semantic_scores.append(semantic)

        # -------------------------
        # MODEL 2 — Fair Ranking
        # FEATURE ORDER MUST MATCH TRAINING
        # -------------------------
        features: List[List[float]] = []
        user_ids: List[str] = []

        for i, out in enumerate(model1_outputs):
            dev = developers.get(submissions[i]["developer_id"], {})

            feature_vector = [
                out["clarity"],
                out["technical_depth"],
                out["creativity"],
                out["overall_quality"],
                sum(out["uncertainty"]) / len(out["uncertainty"]),
                semantic_scores[i],
                float(dev.get("experience", 0)),
                float(dev.get("rating", 0)),
                1.0  # completion proxy (REQUIRED by trained model)
            ]

            features.append(feature_vector)
            user_ids.append(submissions[i]["developer_id"])

        ranked = rank_candidates(
            features=features,
            user_ids=user_ids,
            model_path="ml/artifacts/model_2/fair_ranking.pt"
        )

        ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)

        # -------------------------
        # Enrich for Quantum
        # -------------------------
        enriched = []
        for r in ranked:
            sub = next(s for s in submissions if s["developer_id"] == r["user_id"])
            enriched.append({
                **r,
                "salary": float(sub["expected_pay"]),
                "hours": float(sub["estimated_hours"]),
                "availability": float(sub["availability"]),
            })

        # -------------------------
        # MODEL 3 — Quantum
        # -------------------------
        quantum_candidates = enriched[:50]

        final_user_ids = quantum_select_top_10_percent(
            ranked_user_ids=[c["user_id"] for c in quantum_candidates],
            scores=[c["score"] for c in quantum_candidates],
            salaries=[c["salary"] for c in quantum_candidates],
            hours=[c["hours"] for c in quantum_candidates],
            availability=[c["availability"] for c in quantum_candidates],
        )

        # -------------------------
        # Update DB
        # -------------------------
        for uid in final_user_ids:
            (
                supabase.table("submissions")
                .update({"selected": True})
                .eq("project_id", project_id)
                .eq("developer_id", uid)
                .execute()
            )

        return {
            "shortlisted": final_user_ids,
            "ranking_snapshot": enriched
        }
