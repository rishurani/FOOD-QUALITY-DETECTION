"""
LexBrief AI - Comprehensive Test Suite
Project ID: CSEAIML2026-P13
Tests:
- Legal text preprocessing & citation-safe sentence segmentation
- TextRank & IRAC structured summarization
- Key-element and statute extraction
- Hybrid semantic precedent search & query expansion
- Side-by-side case comparison & concordance analysis
- Flask REST API endpoints and web views
"""
import json
import unittest
from app import app
from legal_nlp.preprocessor import LegalPreprocessor
from legal_nlp.summarizer import LegalSummarizer
from legal_nlp.extractor import LegalElementExtractor
from legal_nlp.semantic_search import SemanticSearchEngine
from legal_nlp.comparator import LegalCaseComparator


class TestLegalNLP(unittest.TestCase):
    """Tests core legal NLP modules."""

    def setUp(self):
        self.sample_judgment = (
            "IN THE SUPREME COURT OF INDIA\n"
            "Civil Appellate Jurisdiction\n"
            "Justice K.S. Puttaswamy (Retd.) v. Union of India\n"
            "Citation: (2017) 10 SCC 1\n"
            "Decided on: August 24, 2017\n"
            "Coram: J.S. Khehar C.J., J. Chelameswar, S.A. Bobde, R.F. Nariman, D.Y. Chandrachud JJ.\n\n"
            "1. The brief facts of the case are that the constitutional validity of the Aadhaar scheme was challenged.\n"
            "2. The substantial question of law for consideration is whether the Right to Privacy is a Fundamental Right guaranteed under Article 21.\n"
            "3. Learned senior counsel for the petitioner contended that privacy is an essential component of human dignity and liberty.\n"
            "4. On the other hand, learned Attorney General for the respondent submitted that the Constitution does not recognize privacy.\n"
            "5. In our considered view, privacy is an inalienable natural right recognized under Part III of the Constitution.\n"
            "6. For the reasons stated above, we hold and declare that the right to privacy is protected under Article 21. Ordered accordingly."
        )

    def test_preprocessor_sentence_splitting(self):
        """Validates that legal abbreviations like v. and Art. do not break sentences."""
        text = "Kesavananda Bharati v. State of Kerala (1973) 4 SCC 225 at p. 250 established the doctrine. The court observed Art. 21."
        sentences = LegalPreprocessor.split_sentences(text)
        self.assertGreaterEqual(len(sentences), 2)
        # Verify 'v.' did not cause premature sentence split
        self.assertTrue(any("Kesavananda Bharati v. State of Kerala" in s for s in sentences))

    def test_preprocessor_stemming(self):
        """Validates legal suffix stemming."""
        self.assertEqual(LegalPreprocessor.stem("constitutional"), LegalPreprocessor.stem("constitution"))
        self.assertEqual(LegalPreprocessor.stem("arguing"), LegalPreprocessor.stem("argued"))

    def test_element_extractor(self):
        """Validates extraction of title, citation, court, outcome, and statutes."""
        elements = LegalElementExtractor.extract_all_elements(self.sample_judgment)
        meta = elements["metadata"]
        self.assertIn("Puttaswamy", meta["title"])
        self.assertIn("Supreme Court of India", meta["court"])
        self.assertEqual(elements["outcome"]["status"], "Allowed")
        self.assertTrue(any("Article 21" in s for s in elements["statutes_cited"]))

    def test_summarizer_irac(self):
        """Validates structured IRAC brief generation and compression ratio."""
        summarizer = LegalSummarizer()
        brief = summarizer.generate_brief(self.sample_judgment, target_sentences=4)
        irac = brief["irac"]
        self.assertTrue(len(irac["facts"]) > 0)
        self.assertTrue(len(irac["issues"]) > 0)
        self.assertTrue(len(irac["ruling"]) > 0)
        self.assertGreater(brief["compression_ratio"], 0.0)

    def test_semantic_search_engine(self):
        """Validates hybrid precedent lookup and relevance scoring."""
        search_engine = SemanticSearchEngine()
        cases = [
            {
                "id": "case-priv",
                "title": "Justice K.S. Puttaswamy v. Union of India",
                "citation": "(2017) 10 SCC 1",
                "court": "Supreme Court of India",
                "domain": "Constitutional Law",
                "issues": ["Right to privacy under Article 21"],
                "ratio_decidendi": "Privacy is protected under Article 21.",
                "statutes_cited": ["Article 21"],
                "text": self.sample_judgment,
                "outcome": {"status": "Allowed", "badge": "success"}
            },
            {
                "id": "case-contract",
                "title": "Hadley v. Baxendale",
                "citation": "(1854) 9 Exch 341",
                "court": "Court of Exchequer",
                "domain": "Contract Law",
                "issues": ["Damages for breach of contract"],
                "ratio_decidendi": "Remoteness of damages in breach.",
                "statutes_cited": ["Contract Act"],
                "text": "The plaintiffs were owners of a steam mill. The crankshaft was broken.",
                "outcome": {"status": "Allowed", "badge": "success"}
            }
        ]
        search_engine.index_cases(cases)
        results = search_engine.search("government surveillance and personal privacy rights", top_k=2)
        self.assertGreater(len(results), 0)
        # Top result must be Puttaswamy
        self.assertEqual(results[0]["case_id"], "case-priv")
        self.assertGreaterEqual(results[0]["score"], 50.0)

    def test_case_comparator(self):
        """Validates side-by-side comparison matrix and concordance analysis."""
        c1 = {
            "id": "c1",
            "title": "Case A",
            "citation": "2020 SC 1",
            "court": "Supreme Court",
            "date": "2020",
            "domain": "Constitutional Law",
            "ratio_decidendi": "Fundamental rights protect personal liberty.",
            "statutes_cited": ["Article 21"],
            "outcome": {"status": "Allowed", "badge": "success"},
            "irac": {"facts": ["F1"], "issues": ["I1"], "arguments": ["A1"], "reasoning": ["R1"], "ruling": ["Allowed"]}
        }
        c2 = {
            "id": "c2",
            "title": "Case B",
            "citation": "2022 SC 2",
            "court": "Supreme Court",
            "date": "2022",
            "domain": "Constitutional Law",
            "ratio_decidendi": "Personal liberty under Article 21 cannot be denied without law.",
            "statutes_cited": ["Article 21"],
            "outcome": {"status": "Allowed", "badge": "success"},
            "irac": {"facts": ["F2"], "issues": ["I2"], "arguments": ["A2"], "reasoning": ["R2"], "ruling": ["Allowed"]}
        }
        result = LegalCaseComparator.compare_cases([c1, c2])
        self.assertIn("matrix", result)
        self.assertIn("analysis", result)
        self.assertTrue(result["analysis"]["same_domain"])


class TestFlaskEndpoints(unittest.TestCase):
    """Tests Flask application routes and REST API."""

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_home_page(self):
        """GET / returns 200 with dashboard content."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'LexBrief', response.data)
        self.assertIn(b'CSEAIML2026-P13', response.data)

    def test_summarize_view(self):
        """GET /summarize returns 200."""
        response = self.client.get('/summarize')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'IRAC', response.data)

    def test_api_summarize_post(self):
        """POST /api/summarize returns structured brief and extracted elements."""
        payload = {
            "text": "The petitioner filed an appeal. The question is whether negligence occurred. We find for the appellant. The appeal is allowed.",
            "target_sentences": 4,
            "mode": "structured"
        }
        response = self.client.post('/api/summarize', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("brief", data)
        self.assertIn("irac", data["brief"])

    def test_api_search_post(self):
        """POST /api/search returns relevant precedent rankings."""
        payload = {
            "query": "duty of care owed by manufacturer to consumer in tort of negligence",
            "top_k": 3
        }
        response = self.client.post('/api/search', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertGreater(data["total_results"], 0)

    def test_api_compare_post(self):
        """POST /api/compare compares two cases by ID."""
        payload = {
            "case_ids": ["puttaswamy-2017", "shreya-singhal-2015"]
        }
        response = self.client.post('/api/compare', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("comparison", data)

    def test_evaluation_view(self):
        """GET /evaluation returns 200 with benchmark metrics."""
        response = self.client.get('/evaluation')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ROUGE', response.data)

    def test_disclaimer_view(self):
        """GET /disclaimer returns 200 with statutory notice."""
        response = self.client.get('/disclaimer')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Not Legal Advice', response.data)


if __name__ == '__main__':
    unittest.main()
