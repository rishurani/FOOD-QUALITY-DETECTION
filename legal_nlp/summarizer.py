"""
Legal Case Summarizer
Implements TextRank / LexRank graph-based extractive summarization with
legal discourse salience weighting, IRAC/FIRAC structured brief generation,
and executive summary synthesis.
"""
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

from .preprocessor import LegalPreprocessor
from .legal_thesaurus import LEGAL_DISCOURSE_MARKERS


class LegalSummarizer:
    """
    Summarizes court judgments into structured legal briefs (Facts, Issues,
    Arguments, Reasoning, Ruling) and concise executive summaries.
    """

    def __init__(self, damping: float = 0.85, max_iter: int = 50, tol: float = 1e-4):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol

    def _sentence_similarity(self, words1: List[str], words2: List[str], idf: Dict[str, float]) -> float:
        """Computes weighted cosine similarity between two sentences."""
        if not words1 or not words2:
            return 0.0

        v1 = Counter(words1)
        v2 = Counter(words2)
        
        # Calculate dot product using TF * IDF
        common_words = set(v1.keys()) & set(v2.keys())
        if not common_words:
            return 0.0

        dot = sum(v1[w] * v2[w] * (idf.get(w, 1.0) ** 2) for w in common_words)
        norm1 = math.sqrt(sum((count * idf.get(w, 1.0)) ** 2 for w, count in v1.items()))
        norm2 = math.sqrt(sum((count * idf.get(w, 1.0)) ** 2 for w, count in v2.items()))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def _compute_sentence_salience_boost(self, sentence: str, position_idx: int, total_sentences: int) -> float:
        """
        Boosts sentences containing strong legal discourse markers or appearing
        in strategically important judgment sections (e.g., final disposition or opening facts).
        """
        s_lower = sentence.lower()
        boost = 1.0

        # Position weight: first 10% (facts setup) and last 15% (holding/ruling) are critical
        rel_pos = position_idx / max(total_sentences, 1)
        if rel_pos <= 0.15:
            boost += 0.35
        elif rel_pos >= 0.85:
            boost += 0.50

        # Legal discourse indicators
        for category, markers in LEGAL_DISCOURSE_MARKERS.items():
            for m in markers:
                if m in s_lower:
                    if category == "ruling":
                        boost += 0.60
                    elif category == "reasoning":
                        boost += 0.45
                    elif category == "issues":
                        boost += 0.40
                    elif "arguments" in category:
                        boost += 0.30
                    break

        # Explicit operative words
        operative_terms = ["held that", "we hold", "we conclude", "accordingly", "ultra vires", "dismissed", "allowed", "ratio"]
        if any(term in s_lower for term in operative_terms):
            boost += 0.40

        return boost

    def _build_similarity_matrix(self, tokenized_sentences: List[List[str]], idf: Dict[str, float]) -> List[List[float]]:
        """Builds pairwise adjacency matrix for sentences."""
        n = len(tokenized_sentences)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._sentence_similarity(tokenized_sentences[i], tokenized_sentences[j], idf)
                if sim > 0.05: # threshold to prune noisy edges
                    matrix[i][j] = sim
                    matrix[j][i] = sim
        return matrix

    def _pagerank(self, matrix: List[List[float]], boosts: List[float]) -> List[float]:
        """Calculates PageRank / TextRank scores with personalization boosts."""
        n = len(matrix)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        # Normalize boost vector for personalized PageRank
        total_boost = sum(boosts) or 1.0
        p_vector = [b / total_boost for b in boosts]

        # Row normalization of weights
        norm_matrix = []
        for i in range(n):
            row_sum = sum(matrix[i])
            if row_sum > 0:
                norm_matrix.append([matrix[i][j] / row_sum for j in range(n)])
            else:
                norm_matrix.append([1.0 / n for _ in range(n)])

        # Power iteration
        scores = list(p_vector)
        for _ in range(self.max_iter):
            new_scores = [0.0] * n
            for i in range(n):
                rank_sum = sum(scores[j] * norm_matrix[j][i] for j in range(n))
                new_scores[i] = (1.0 - self.damping) * p_vector[i] + self.damping * rank_sum

            # Check convergence
            diff = sum(abs(new_scores[i] - scores[i]) for i in range(n))
            scores = new_scores
            if diff < self.tol:
                break

        return scores

    def classify_sentence_zone(self, sentence: str, position_idx: int, total_sentences: int) -> str:
        """Classifies a sentence into one of the IRAC zones."""
        s_lower = sentence.lower()
        rel_pos = position_idx / max(total_sentences, 1)

        # 1. Ruling (mostly in concluding sections or explicit holding)
        ruling_markers = LEGAL_DISCOURSE_MARKERS["ruling"] + [
            "appeal is allowed", "appeal is dismissed", "writ petition is allowed",
            "we hold that", "conviction is set aside", "order of the high court is",
            "ordered accordingly", "direction is issued"
        ]
        if any(m in s_lower for m in ruling_markers) and rel_pos >= 0.5:
            return "ruling"

        # 2. Issues
        issue_markers = LEGAL_DISCOURSE_MARKERS["issues"] + [
            "the question arises", "whether the", "point for consideration", "issue to be decided"
        ]
        if any(m in s_lower for m in issue_markers):
            return "issues"

        # 3. Arguments
        petitioner_markers = LEGAL_DISCOURSE_MARKERS["arguments_petitioner"]
        respondent_markers = LEGAL_DISCOURSE_MARKERS["arguments_respondent"]
        if any(m in s_lower for m in petitioner_markers):
            return "arguments_petitioner"
        if any(m in s_lower for m in respondent_markers):
            return "arguments_respondent"

        # 4. Reasoning
        reasoning_markers = LEGAL_DISCOURSE_MARKERS["reasoning"] + [
            "in our view", "we are of the opinion", "principle laid down", "it is clear that",
            "we cannot accept", "the doctrine of", "ratio decidendi", "precedent"
        ]
        if any(m in s_lower for m in reasoning_markers):
            return "reasoning"

        # Fallback based on relative position in the judgment
        if rel_pos <= 0.30:
            return "facts"
        elif rel_pos <= 0.70:
            return "reasoning"
        else:
            return "ruling"

    def generate_brief(
        self,
        text: str,
        target_sentences: int = 8,
        mode: str = "structured",
        focus: str = "all"
    ) -> Dict[str, Any]:
        """
        Main entry point: Generates a structured brief from court judgment text.
        
        Args:
            text: Full text of court judgment
            target_sentences: Target sentence count for summary
            mode: 'structured' (IRAC), 'executive' (narrative), or 'bullet' (takeaways)
            focus: 'all', 'facts', 'reasoning', or 'ruling'
        """
        sentences = LegalPreprocessor.split_sentences(text)
        if not sentences:
            return {
                "executive_summary": "No text provided for summarization.",
                "irac": {"facts": [], "issues": [], "arguments": [], "reasoning": [], "ruling": []},
                "key_takeaways": [],
                "compression_ratio": 0.0,
                "stats": {"total_words": 0, "summary_words": 0, "total_sentences": 0, "summary_sentences": 0}
            }

        total_words = sum(len(s.split()) for s in sentences)
        n = len(sentences)

        # Tokenize and compute IDF
        tokenized = [LegalPreprocessor.tokenize(s, remove_stopwords=True) for s in sentences]
        doc_freq = Counter()
        for tok_set in [set(t) for t in tokenized]:
            doc_freq.update(tok_set)

        idf = {word: math.log((n + 1) / (freq + 1)) + 1.0 for word, freq in doc_freq.items()}

        # Build Graph and calculate PageRank with Legal Boosts
        boosts = [self._compute_sentence_salience_boost(sentences[i], i, n) for i in range(n)]
        matrix = self._build_similarity_matrix(tokenized, idf)
        scores = self._pagerank(matrix, boosts)

        # Categorize all sentences into IRAC zones with scores
        categorized = defaultdict(list)
        for idx, (sent, score) in enumerate(zip(sentences, scores)):
            zone = self.classify_sentence_zone(sent, idx, n)
            categorized[zone].append((idx, sent, score))

        # Select top sentences per IRAC zone for structured brief
        # Facts (1-3 sentences)
        facts_candidates = sorted(categorized["facts"], key=lambda x: x[2], reverse=True)
        facts = [x[1] for x in sorted(facts_candidates[:2], key=lambda x: x[0])]

        # Issues (1-2 sentences)
        issues_candidates = sorted(categorized["issues"], key=lambda x: x[2], reverse=True)
        issues = [x[1] for x in sorted(issues_candidates[:2], key=lambda x: x[0])]

        # Arguments (Petitioner & Respondent)
        arg_p_candidates = sorted(categorized["arguments_petitioner"], key=lambda x: x[2], reverse=True)
        arg_r_candidates = sorted(categorized["arguments_respondent"], key=lambda x: x[2], reverse=True)
        arguments = []
        if arg_p_candidates:
            arguments.append(f"Petitioner/Appellant Contention: {arg_p_candidates[0][1]}")
        if arg_r_candidates:
            arguments.append(f"Respondent/State Contention: {arg_r_candidates[0][1]}")
        if not arguments and (arg_p_candidates or arg_r_candidates):
            arguments = [x[1] for x in sorted((arg_p_candidates + arg_r_candidates)[:2], key=lambda x: x[0])]

        # Reasoning (2-4 sentences)
        reasoning_candidates = sorted(categorized["reasoning"], key=lambda x: x[2], reverse=True)
        reasoning = [x[1] for x in sorted(reasoning_candidates[:3], key=lambda x: x[0])]

        # Ruling (1-2 sentences)
        ruling_candidates = sorted(categorized["ruling"], key=lambda x: x[2], reverse=True)
        ruling = [x[1] for x in sorted(ruling_candidates[:2], key=lambda x: x[0])]

        # Fallbacks if some categories are empty
        if not facts and sentences:
            facts = [sentences[0]]
        if not ruling and len(sentences) > 1:
            ruling = [sentences[-1]]
        if not reasoning and len(sentences) > 2:
            reasoning = [sentences[len(sentences) // 2]]

        # Executive Summary: Top ranked sentences maintaining chronological flow
        ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
        num_exec = min(max(target_sentences, 4), n)
        top_indices = sorted(ranked_indices[:num_exec])
        exec_sentences = [sentences[i] for i in top_indices]
        executive_summary = " ".join(exec_sentences)

        # Key Takeaways: High impact bullet points
        takeaways = []
        if ruling:
            takeaways.append(f"Holding: {ruling[0]}")
        if reasoning:
            takeaways.append(f"Key Principle: {reasoning[0]}")
        if issues:
            takeaways.append(f"Legal Question: {issues[0]}")
        elif len(exec_sentences) > 1:
            takeaways.append(f"Core Finding: {exec_sentences[1]}")

        summary_words = sum(len(s.split()) for s in exec_sentences)
        compression_ratio = round((1.0 - (summary_words / max(total_words, 1))) * 100, 1)

        return {
            "executive_summary": executive_summary,
            "irac": {
                "facts": facts,
                "issues": issues,
                "arguments": arguments,
                "reasoning": reasoning,
                "ruling": ruling
            },
            "key_takeaways": takeaways,
            "compression_ratio": max(0.0, compression_ratio),
            "stats": {
                "total_words": total_words,
                "summary_words": summary_words,
                "total_sentences": n,
                "summary_sentences": len(exec_sentences)
            }
        }
