"""
Legal Case Comparator
Provides side-by-side comparative analysis of court judgments,
identifying doctrinal concordance, distinction, or conflicts across cases.
"""
from typing import List, Dict, Any
from .preprocessor import LegalPreprocessor


class LegalCaseComparator:
    """Compares multiple court judgments side by side across IRAC dimensions."""

    @staticmethod
    def _compute_jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
        """Computes Jaccard similarity between two token sets."""
        s1 = set(tokens1)
        s2 = set(tokens2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    @classmethod
    def compare_cases(cls, cases_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds a comprehensive comparison matrix for 2 or more cases.
        
        Args:
            cases_list: List of parsed case dictionary objects
        """
        if len(cases_list) < 2:
            return {"error": "At least two cases are required for comparative analysis."}

        # Build comparison columns
        comparison_matrix = {
            "metadata": [],
            "facts": [],
            "issues": [],
            "arguments": [],
            "reasoning": [],
            "statutes": [],
            "rulings": []
        }

        for c in cases_list:
            irac = c.get("irac", {})
            comparison_matrix["metadata"].append({
                "id": c.get("id"),
                "title": c.get("title"),
                "citation": c.get("citation"),
                "court": c.get("court"),
                "date": c.get("date"),
                "domain": c.get("domain"),
                "outcome": c.get("outcome", {})
            })
            comparison_matrix["facts"].append({
                "case_id": c.get("id"),
                "title": c.get("title"),
                "content": irac.get("facts", ["No facts recorded."])
            })
            comparison_matrix["issues"].append({
                "case_id": c.get("id"),
                "title": c.get("title"),
                "content": irac.get("issues", c.get("issues", ["General legal question"]))
            })
            comparison_matrix["arguments"].append({
                "case_id": c.get("id"),
                "title": c.get("title"),
                "content": irac.get("arguments", ["Arguments documented in brief."])
            })
            comparison_matrix["reasoning"].append({
                "case_id": c.get("id"),
                "title": c.get("title"),
                "content": c.get("ratio_decidendi", "") or " ".join(irac.get("reasoning", []))
            })
            comparison_matrix["statutes"].append({
                "case_id": c.get("id"),
                "title": c.get("title"),
                "content": c.get("statutes_cited", [])
            })
            comparison_matrix["rulings"].append({
                "case_id": c.get("id"),
                "title": c.get("title"),
                "content": irac.get("ruling", [c.get("outcome", {}).get("summary", "")])
            })

        # Concordance / Conflict Analysis between Case 1 and Case 2
        c1 = cases_list[0]
        c2 = cases_list[1]
        
        r1_text = c1.get("ratio_decidendi", "") + " " + " ".join(c1.get("irac", {}).get("reasoning", []))
        r2_text = c2.get("ratio_decidendi", "") + " " + " ".join(c2.get("irac", {}).get("reasoning", []))
        
        t1 = LegalPreprocessor.tokenize(r1_text, remove_stopwords=True)
        t2 = LegalPreprocessor.tokenize(r2_text, remove_stopwords=True)
        
        sim = cls._compute_jaccard_similarity(t1, t2)
        sim_pct = round(sim * 100 * 2.5, 1) # scaled for presentation
        sim_pct = min(sim_pct, 95.0)

        shared_statutes = list(set(c1.get("statutes_cited", [])) & set(c2.get("statutes_cited", [])))
        same_domain = (c1.get("domain", "") == c2.get("domain", ""))
        
        # Determine relationship
        out1 = c1.get("outcome", {}).get("status", "")
        out2 = c2.get("outcome", {}).get("status", "")

        if same_domain and sim_pct >= 40.0:
            if out1 == out2:
                relationship = "Concordant / Precedent Followed"
                rel_badge = "success"
                rel_desc = (
                    f"Both cases share substantial doctrinal alignment in {c1.get('domain')}. "
                    f"The legal reasoning in '{c2.get('title')}' reinforces and follows the principles established in '{c1.get('title')}'."
                )
            else:
                relationship = "Distinguished on Factual Matrix"
                rel_badge = "warning"
                rel_desc = (
                    f"While both cases address {c1.get('domain')}, distinct factual matrices and evidentiary thresholds "
                    f"led the bench to distinguish '{c2.get('title')}' from '{c1.get('title')}'."
                )
        elif same_domain and out1 != out2 and sim_pct < 40.0:
            relationship = "Divergent Doctrines / Distinct Categories"
            rel_badge = "danger"
            rel_desc = (
                f"The cases adopt diverging judicial approaches within {c1.get('domain')}. "
                f"One judgment imposes strict liability/conviction while the other limits scope or dismisses."
            )
        else:
            relationship = "Comparative Across Distinct Legal Domains"
            rel_badge = "info"
            rel_desc = (
                f"Comparison highlights distinct jurisprudential areas ({c1.get('domain')} vs {c2.get('domain')}), "
                f"illustrating how procedural due process operates across different legal contexts."
            )

        return {
            "matrix": comparison_matrix,
            "analysis": {
                "similarity_score": sim_pct,
                "relationship": relationship,
                "relationship_badge": rel_badge,
                "description": rel_desc,
                "shared_statutes": shared_statutes,
                "same_domain": same_domain
            }
        }
