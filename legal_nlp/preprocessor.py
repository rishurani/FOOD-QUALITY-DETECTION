"""
Legal Text Preprocessor
Handles sentence tokenization, legal abbreviation protection, cleaning,
and specialized legal stopword handling.
"""
import re
import string
from typing import List, Tuple

# Protected legal abbreviations that should not trigger sentence boundaries
LEGAL_ABBREVIATIONS = [
    r'v\.', r'vs\.', r'art\.', r'sec\.', r'para\.', r'hon\'ble', r'no\.', r'ltd\.',
    r'co\.', r'inc\.', r'corp\.', r'govt\.', r'ors\.', r'anr\.', r'scc\.', r'air\.',
    r'scr\.', r'u/s\.', r'u/s', r'cr\.p\.c\.', r'i\.p\.c\.', r'c\.p\.c\.', r'f\.i\.r\.',
    r'd\.b\.', r'c\.j\.', r'j\.', r'jj\.', r'dr\.', r'mr\.', r'mrs\.', r'ms\.',
    r'prof\.', r'dept\.', r'ed\.', r'vol\.', r'pp\.', r'p\.', r'ibid\.', r'id\.',
    r'op\. cit\.', r'loc\. cit\.', r'viz\.', r'i\.e\.', r'e\.g\.', r'etc\.'
]

# Standard English stopwords (excluding operative legal words like 'shall', 'liable', 'breach', 'guilty', 'against')
LEGAL_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'all', 'am', 'an', 'and', 'any', 'are',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both',
    'but', 'by', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
    'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself',
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so',
    'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
    'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself',
    'yourselves'
}


class LegalPreprocessor:
    """Preprocesses and normalizes legal judgment texts and queries."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Removes extraneous formatting, replaces curly quotes, and standardizes spacing."""
        if not text:
            return ""
        # Normalize quotes and dashes
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        text = text.replace('—', ' - ').replace('–', ' - ')
        # Collapse multiple newlines and spaces
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """
        Splits legal text into sentences while protecting citations,
        section references, abbreviations, and list numbers.
        """
        cleaned = cls.clean_text(text)
        if not cleaned:
            return []

        # Mask abbreviations with a placeholder
        masked = cleaned
        masks = {}
        for idx, abbr in enumerate(LEGAL_ABBREVIATIONS):
            matches = list(re.finditer(abbr, masked, flags=re.IGNORECASE))
            for m in reversed(matches):
                key = f"__LEGAL_ABBR_{idx}_{m.start()}__"
                masks[key] = m.group(0)
                masked = masked[:m.start()] + key + masked[m.end():]

        # Protect citation patterns like (2017) 10 SCC 1 or [1932] AC 562
        citation_matches = list(re.finditer(r'[\(\[]\d{4}[\)\]]\s+\d+\s+[A-Za-z\.]+\s+\d+', masked))
        for idx, m in enumerate(reversed(citation_matches)):
            key = f"__LEGAL_CIT_{idx}__"
            masks[key] = m.group(0)
            masked = masked[:m.start()] + key + masked[m.end():]

        # Protect numbered list headers e.g. "1.", "(a)", "ii."
        list_matches = list(re.finditer(r'(?:^|\n)\s*(?:\d+\.|\([a-zA-Z0-9]+\))\s*', masked))
        for idx, m in enumerate(reversed(list_matches)):
            key = f"__LEGAL_LIST_{idx}__"
            masks[key] = m.group(0)
            masked = masked[:m.start()] + key + masked[m.end():]

        # Sentence split on punctuation followed by whitespace and capital letter
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘])', masked)

        # Restore placeholders and filter
        final_sentences = []
        for s in raw_sentences:
            s_clean = s
            for key, orig in masks.items():
                s_clean = s_clean.replace(key, orig)
            s_clean = s_clean.strip()
            # Must have at least 15 characters and at least 3 words to be considered a legal sentence
            if len(s_clean) >= 15 and len(s_clean.split()) >= 3:
                final_sentences.append(s_clean)

        return final_sentences

    @classmethod
    def tokenize(cls, text: str, lower: bool = True, remove_stopwords: bool = True) -> List[str]:
        """Tokenizes text into words, handling legal symbols."""
        if lower:
            text = text.lower()
        # Remove punctuation except internal hyphens
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b', text)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in LEGAL_STOPWORDS]
        return tokens

    @classmethod
    def stem(cls, word: str) -> str:
        """
        Lightweight Porter-like suffix stemmer tuned for legal English terms.
        Handles words like: argues/argued/argument -> argu, constitutional -> constitut
        """
        w = word.lower()
        if len(w) <= 3:
            return w
        # Common legal suffixes
        suffixes = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
            ('izer', 'ize'), ('abli', 'able'), ('alli', 'al'), ('entli', 'ent'),
            ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
            ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
            ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),
            ('ing', ''), ('ed', ''), ('ies', 'y'), ('es', ''), ('s', ''), ('ment', '')
        ]
        for suf, rep in suffixes:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[:-len(suf)] + rep
        return w

    @classmethod
    def get_ngrams(cls, tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
        """Extracts n-grams for phrase-level matching (e.g. 'duty of care', 'habeas corpus')."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
