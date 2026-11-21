"""
Legal Element Extractor
Extracts case metadata, facts, arguments, ratio decidendi, final ruling,
statutes cited, precedents cited, and legal domain classifications.
"""
import re
from typing import Dict, List, Any

from .preprocessor import LegalPreprocessor
from .legal_thesaurus import LEGAL_ONTOLOGY


class LegalElementExtractor:
    """Extracts structured legal entities and elements from court judgment texts."""

    @staticmethod
    def extract_metadata(text: str) -> Dict[str, Any]:
        """Extracts Title, Citation, Court, Bench, Date, and Parties."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        header = "\n".join(lines[:15]) if lines else ""

        # Title pattern (Party A v./vs. Party B)
        title_match = re.search(
            r'([A-Z][A-Za-z0-9\.,\s\(\)\'\-&]+?\s+(?:v\.|vs\.|versus)\s+[A-Za-z0-9\.,\s\(\)\'\-&]+)',
            header,
            re.IGNORECASE
        )
        title = title_match.group(1).strip() if title_match else (lines[0] if lines else "Unknown Case")

        # Citation pattern (e.g., (2017) 10 SCC 1, AIR 1973 SC 1461, 384 U.S. 436, [1932] AC 562)
        citation_match = re.search(
            r'(?:(?:[\(\[]\d{4}[\)\]]|\d{4})\s+\d*\s*(?:SCC|AIR|SCR|U\.S\.|AC|All ER|F\.3d|SCC OnLine)\s+\d+(?:[\s\w]*))',
            header,
            re.IGNORECASE
        )
        citation = citation_match.group(0).strip() if citation_match else "Citation Pending / Unreported"

        # Court Name
        court = "Apex Court / High Court"
        court_patterns = [
            (r'Supreme Court of India', 'Supreme Court of India'),
            (r'Supreme Court of the United States', 'Supreme Court of the United States'),
            (r'House of Lords', 'House of Lords (UK)'),
            (r'High Court of [A-Za-z\s]+', None),
            (r'Court of Appeal', 'Court of Appeal'),
            (r'Supreme Court', 'Supreme Court')
        ]
        for pat, name in court_patterns:
            m = re.search(pat, header, re.IGNORECASE)
            if m:
                court = name if name else m.group(0).strip()
                break

        # Bench / Judges
        bench = []
        bench_matches = re.findall(
            r'(?:Hon\'ble\s+)?(?:Mr\.|Mrs\.|Justice|Chief Justice|J\.|CJ)\s+([A-Z][a-zA-Z\s\.]+?)(?:,\s*|\sand\s|\n|\.$)',
            header
        )
        if bench_matches:
            bench = [b.strip() for b in bench_matches if len(b.strip()) > 3][:5]
        if not bench:
            bench = ["Coram details in record"]

        # Date of Decision
        date_match = re.search(
            r'(?:Decided on|Date of Judgment|Judgment dated|Dated|Decided):\s*([0-9]{1,2}[\s\/\-][A-Za-z0-9]+[\s\/\-][0-9]{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4})',
            header,
            re.IGNORECASE
        )
        date = date_match.group(1).strip() if date_match else "Date in record"

        return {
            "title": title,
            "citation": citation,
            "court": court,
            "bench": bench,
            "date": date
        }

    @staticmethod
    def extract_statutes(text: str) -> List[str]:
        """Extracts cited statutory provisions, articles, sections, and acts."""
        statutes = set()

        # Constitutional Articles (e.g. Article 21, Art. 14, Article 19(1)(a))
        art_matches = re.findall(
            r'(?:Article|Art\.)\s+\d+[A-Z]?(?:\(\d+\)(?:\([a-z]\))?)?(?:\s+of\s+the\s+Constitution(?:\s+of\s+India)?)?',
            text,
            re.IGNORECASE
        )
        for m in art_matches:
            statutes.add(m.strip())

        # Statutory Sections (e.g. Section 302 of the Indian Penal Code, Sec. 66A IT Act, Section 73)
        sec_matches = re.findall(
            r'(?:Section|Sec\.)\s+\d+[A-Z]?(?:\s+(?:of\s+(?:the\s+)?|u/s\s+)?(?:[A-Z][A-Za-z\s]+(?:Act|Code|Rules)))?',
            text,
            re.IGNORECASE
        )
        for m in sec_matches:
            clean_m = re.sub(r'\s+', ' ', m).strip()
            if len(clean_m) > 7:
                statutes.add(clean_m)

        # Common Acts
        act_matches = re.findall(
            r'(?:Indian Penal Code|Code of Criminal Procedure|Civil Procedure Code|Information Technology Act|Patents Act|Contract Act|Evidence Act|Constitution of India)',
            text,
            re.IGNORECASE
        )
        for m in act_matches:
            statutes.add(m.strip())

        return sorted(list(statutes))[:12]

    @staticmethod
    def extract_precedents(text: str) -> List[str]:
        """Extracts cited precedents / case law references from judgment text."""
        precedents = set()

        # Named cases (e.g., Donoghue v. Stevenson, Kesavananda Bharati v. State)
        case_matches = re.findall(
            r'([A-Z][a-zA-Z\.\s\'-]{2,25}\s+(?:v\.|vs\.|versus)\s+[A-Z][a-zA-Z\.\s\'-]{2,25}(?:,\s*[\(\[]\d{4}[\)\]]\s+[A-Za-z0-9\s\.]+)?\b)',
            text
        )
        for m in case_matches:
            cand = m.strip()
            # Filter noise
            if not any(stop in cand.lower() for stop in ['learned counsel', 'appeal filed by', 'affidavit of']):
                precedents.add(cand)

        return sorted(list(precedents))[:10]

    @staticmethod
    def detect_outcome(text: str) -> Dict[str, str]:
        """Detects whether the appeal/petition was allowed, dismissed, acquitted, remanded, etc."""
        text_lower = text[-2500:].lower() # inspect final 2500 chars

        if any(term in text_lower for term in ['appeal is allowed', 'appeals are allowed', 'writ petition is allowed', 'petition allowed']):
            return {"status": "Allowed", "badge": "success", "color": "#10b981", "summary": "The court ruled in favor of the appellant / petitioner."}
        elif any(term in text_lower for term in ['appeal is dismissed', 'appeals are dismissed', 'writ petition is dismissed', 'petition dismissed']):
            return {"status": "Dismissed", "badge": "danger", "color": "#ef4444", "summary": "The court dismissed the claims, upholding the lower court decision."}
        elif any(term in text_lower for term in ['conviction is set aside', 'acquitted of all charges', 'acquittal']):
            return {"status": "Acquitted", "badge": "info", "color": "#06b6d4", "summary": "Conviction was set aside and the accused was acquitted."}
        elif any(term in text_lower for term in ['remanded back', 'remit the matter', 'remanded to']):
            return {"status": "Remanded", "badge": "warning", "color": "#f59e0b", "summary": "Matter remanded back to lower authority/court for reconsideration."}
        elif any(term in text_lower for term in ['partly allowed', 'partially allowed']):
            return {"status": "Partly Allowed", "badge": "warning", "color": "#8b5cf6", "summary": "Appeal or petition was partially granted with modified relief."}
        else:
            return {"status": "Disposed / Direction Issued", "badge": "secondary", "color": "#64748b", "summary": "Judgment delivered with binding declarations and directives."}

    @staticmethod
    def detect_legal_domain(text: str) -> str:
        """Classifies judgment into legal domain based on concept ontology matches."""
        text_lower = text.lower()
        domain_scores = {}

        for domain, data in LEGAL_ONTOLOGY.items():
            score = 0
            for concept in data.get("concepts", []):
                if concept in text_lower:
                    score += 2
            for syn_group in data.get("synonyms", {}).values():
                for syn in syn_group:
                    if syn in text_lower:
                        score += 1
            domain_scores[domain] = score

        best_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
        domain_names = {
            "constitutional_law": "Constitutional Law",
            "criminal_law": "Criminal Law & Procedure",
            "contract_law": "Contract & Commercial Law",
            "tort_law": "Tort & Negligence Law",
            "intellectual_property": "Intellectual Property Rights",
            "environmental_law": "Environmental Law"
        }
        return domain_names.get(best_domain, "General Law")

    @classmethod
    def extract_all_elements(cls, text: str) -> Dict[str, Any]:
        """Runs the complete extraction pipeline on raw judgment text."""
        metadata = cls.extract_metadata(text)
        statutes = cls.extract_statutes(text)
        precedents = cls.extract_precedents(text)
        outcome = cls.detect_outcome(text)
        domain = cls.detect_legal_domain(text)

        return {
            "metadata": metadata,
            "domain": domain,
            "statutes_cited": statutes,
            "precedents_cited": precedents,
            "outcome": outcome
        }
