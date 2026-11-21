"""
Semantic Search & Precedent Retrieval Engine
Implements hybrid retrieval: BM25 keyword matching + Dense Legal Semantic Vector
Cosine Similarity + Query Expansion and relevance explanations.
"""
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

from .preprocessor import LegalPreprocessor
from .legal_thesaurus import expand_query, LEGAL_ONTOLOGY


class SemanticSearchEngine:
    """
    Search engine for court case database supporting natural-language queries,
    semantic vector retrieval, concept matching, and hybrid scoring.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, semantic_weight: float = 0.65):
        self.k1 = k1
        self.b = b
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.cases: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avg_dl: float = 0.0
        self.df: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.vocabulary: List[str] = []
        self.case_vectors: List[Dict[str, float]] = []

    def index_cases(self, cases: List[Dict[str, Any]]) -> None:
        """Indexes a list of court cases for semantic and BM25 search."""
        self.cases = cases
        n = len(cases)
        if n == 0:
            return

        self.doc_tokens = []
        self.doc_lens = []
        self.df = Counter()

        for case in cases:
            # Concatenate title, issues, ratio, domain, and full text with weighting
            composite_text = (
                (case.get("title", "") + " ") * 3 +
                (case.get("domain", "") + " ") * 2 +
                (" ".join(case.get("issues", [])) + " ") * 2 +
                (case.get("ratio_decidendi", "") + " ") * 3 +
                (" ".join(case.get("statutes_cited", [])) + " ") * 2 +
                case.get("text", "")
            )
            tokens = LegalPreprocessor.tokenize(composite_text, remove_stopwords=True)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            self.df.update(set(tokens))

        self.avg_dl = sum(self.doc_lens) / max(n, 1)

        # Compute IDF
        self.idf = {}
        for term, freq in self.df.items():
            # BM25 IDF formulation
            self.idf[term] = math.log(((n - freq + 0.5) / (freq + 0.5)) + 1.0)

        # Precompute document vectors for fast cosine similarity
        self.case_vectors = []
        for i, tokens in enumerate(self.doc_tokens):
            vec = self._compute_tf_idf_vector(tokens)
            # Add domain concept bonus to document vector
            case_text_lower = (cases[i].get("text", "") + " " + cases[i].get("ratio_decidendi", "")).lower()
            for domain, data in LEGAL_ONTOLOGY.items():
                for concept in data.get("concepts", []):
                    if concept in case_text_lower:
                        concept_token = concept.replace(" ", "_")
                        vec[concept_token] = vec.get(concept_token, 0.0) + 1.5
            self.case_vectors.append(vec)

    def _compute_tf_idf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Calculates sublinear TF-IDF vector with Euclidean unit normalization."""
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            if term in self.idf:
                # Sublinear term frequency (1 + log(tf)) * idf
                sublinear_tf = 1.0 + math.log(count)
                vec[term] = sublinear_tf * self.idf[term]

        # Normalize to unit length
        norm = math.sqrt(sum(val ** 2 for val in vec.values()))
        if norm > 0:
            for term in vec:
                vec[term] /= norm
        return vec

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Computes cosine similarity between two sparse vectors."""
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0
        return sum(vec1[k] * vec2[k] for k in common_keys)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Computes Okapi BM25 score for a document."""
        score = 0.0
        doc_tokens = self.doc_tokens[doc_idx]
        tf = Counter(doc_tokens)
        dl = self.doc_lens[doc_idx]

        for q in query_tokens:
            if q not in self.idf:
                continue
            term_freq = tf.get(q, 0)
            if term_freq == 0:
                continue
            numerator = self.idf[q] * term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * (dl / max(self.avg_dl, 1.0)))
            score += numerator / denominator

        return score

    def _find_matching_concepts(self, query: str, case: Dict[str, Any]) -> List[str]:
        """Identifies shared legal concepts between user query and case."""
        q_lower = query.lower()
        case_text = (case.get("title", "") + " " + case.get("ratio_decidendi", "") + " " + " ".join(case.get("issues", []))).lower()
        matched = []

        for domain, data in LEGAL_ONTOLOGY.items():
            for concept in data.get("concepts", []):
                # Check if concept words appear in both query and case
                concept_words = [w for w in concept.split() if len(w) > 3]
                if any(w in q_lower for w in concept_words) and concept in case_text:
                    matched.append(concept)
            for key, syns in data.get("synonyms", {}).items():
                if key in q_lower or any(s in q_lower for s in syns):
                    if key in case_text or any(s in case_text for s in syns):
                        matched.append(key)

        return sorted(list(set(matched)))[:4]

    def _extract_best_excerpt(self, query_tokens: List[str], text: str) -> str:
        """Finds the most relevant sentence in the text matching query terms."""
        sentences = LegalPreprocessor.split_sentences(text)
        if not sentences:
            return ""

        q_stems = {LegalPreprocessor.stem(t) for t in query_tokens}
        best_sentence = sentences[0]
        max_matches = -1

        for s in sentences:
            s_tokens = LegalPreprocessor.tokenize(s, remove_stopwords=True)
            s_stems = {LegalPreprocessor.stem(t) for t in s_tokens}
            overlap = len(q_stems & s_stems)
            if overlap > max_matches:
                max_matches = overlap
                best_sentence = s

        return best_sentence

    def search(
        self,
        query: str,
        domain_filter: Optional[str] = None,
        outcome_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid semantic search on legal database.
        
        Args:
            query: Natural language research query
            domain_filter: Optional legal domain filter
            outcome_filter: Optional outcome filter
            top_k: Maximum number of results to return
        """
        if not query.strip() or not self.cases:
            return []

        # 1. Expand query with legal synonyms
        expanded_query = expand_query(query)
        query_tokens = LegalPreprocessor.tokenize(query, remove_stopwords=True)
        expanded_tokens = LegalPreprocessor.tokenize(expanded_query, remove_stopwords=True)

        # 2. Build query vector
        query_vec = self._compute_tf_idf_vector(expanded_tokens)
        # Add concept weights to query vector
        q_lower = query.lower()
        for domain, data in LEGAL_ONTOLOGY.items():
            for concept in data.get("concepts", []):
                if concept in q_lower:
                    concept_token = concept.replace(" ", "_")
                    query_vec[concept_token] = query_vec.get(concept_token, 0.0) + 1.5

        # 3. Compute hybrid scores
        scores = []
        bm25_raw = [self._bm25_score(query_tokens, i) for i in range(len(self.cases))]
        max_bm25 = max(bm25_raw) if bm25_raw and max(bm25_raw) > 0 else 1.0

        for i, case in enumerate(self.cases):
            # Apply domain filter
            if domain_filter and domain_filter != "all":
                if case.get("domain", "").lower() != domain_filter.lower():
                    continue

            # Apply outcome filter
            if outcome_filter and outcome_filter != "all":
                case_outcome = case.get("outcome", {}).get("status", "").lower()
                if case_outcome != outcome_filter.lower():
                    continue

            # Semantic Vector Cosine Similarity
            sem_sim = self._cosine_similarity(query_vec, self.case_vectors[i])

            # Normalized BM25 score
            bm25_norm = bm25_raw[i] / max_bm25 if max_bm25 > 0 else 0.0

            # Composite Hybrid Score
            composite_score = (self.semantic_weight * sem_sim) + (self.keyword_weight * bm25_norm)

            # Match explanation
            matched_concepts = self._find_matching_concepts(query, case)
            if matched_concepts:
                composite_score += 0.08 * len(matched_concepts)

            # Snippet excerpt
            excerpt = self._extract_best_excerpt(query_tokens, case.get("text", ""))

            scores.append({
                "case_id": case.get("id"),
                "title": case.get("title"),
                "citation": case.get("citation"),
                "court": case.get("court"),
                "date": case.get("date"),
                "domain": case.get("domain"),
                "outcome": case.get("outcome"),
                "ratio_decidendi": case.get("ratio_decidendi", ""),
                "score": round(min(composite_score * 100, 99.8), 1),
                "semantic_score": round(sem_sim * 100, 1),
                "keyword_score": round(bm25_norm * 100, 1),
                "matched_concepts": matched_concepts,
                "relevant_excerpt": excerpt,
                "relevance_explanation": (
                    f"Matched {len(matched_concepts)} core legal concepts ({', '.join(matched_concepts)}) "
                    f"with {round(sem_sim * 100)}% legal reasoning semantic alignment."
                    if matched_concepts else
                    f"Semantic vector alignment on jurisprudence principles ({round(sem_sim * 100)}% match)."
                )
            })

        # Sort by composite score descending
        ranked = sorted(scores, key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
