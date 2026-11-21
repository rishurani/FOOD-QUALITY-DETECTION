# LexBrief AI: AI-Based Court Case Summarization & Legal Research Assistant
**Project ID:** `CSEAIML2026-P13`  
**Specialization:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
**Primary Users:** Legal professionals, paralegals, law students, and self-represented litigants  

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0-success.svg)](https://flask.palletsprojects.com/)
[![ROUGE-1 Score](https://img.shields.io/badge/ROUGE--1%20F1-49.6%25-brightgreen.svg)]()
[![Precision@1](https://img.shields.io/badge/Precision%401-100%25-gold.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

---

## 1. Project Overview & Problem Statement

Lengthy court judgments, often spanning 50 to 200 pages, present severe bottlenecks in the administration of justice. Advocates, judicial law clerks, and researchers spend countless hours manually distilling procedural histories, legal issues, opposing arguments, and the *ratio decidendi* (binding legal principle). Moreover, traditional keyword-based case law repositories (e.g. searching for exact statutory strings) fail when legal principles are articulated through varying conceptual vocabulary.

**LexBrief AI** is a specialized Natural Language Processing system engineered to address these challenges:
1. **Condensation into Structured IRAC Briefs:** Employs an extractive graph-centrality algorithm (*TextRank/LexRank*) augmented with legal discourse salience priors to segment judgments into **Facts, Issues, Arguments, Reasoning, and Rulings**.
2. **Automated Key-Element Extraction:** Isolates judicial coram, decision disposition badges (Allowed, Dismissed, Acquitted), cited constitutional provisions, statutory sections, and precedent cross-references.
3. **Semantic Precedent Search Engine:** Implements a hybrid information retrieval model combining **BM25 keyword matching** with **dense legal ontology concept embeddings** and query expansion.
4. **Side-by-Side Multi-Case Comparison Interface:** Allows users to compare two or more judgments across IRAC dimensions, computing doctrinal concordance and distinguishing factors.
5. **Rigorous Ethical AI Governance:** Features persistent legal disclaimers and scope-of-use policies emphasizing that outputs support research and do not substitute for licensed legal advice.

---

## 2. System Architecture

```
                                      +---------------------------------------------+
                                      |             Court Judgment Input            |
                                      | (Uploaded .txt, Raw Text, or Pre-Indexed DB)|
                                      +---------------------------------------------+
                                                             |
                                                             v
                                      +---------------------------------------------+
                                      |       Legal Preprocessing & Segmenter       |
                                      |  - Citation-Safe Sentence Splitter          |
                                      |  - Legal Abbreviation Masking (v., Art., JJ)|
                                      |  - Legal Stopword Filtering & Stemmer       |
                                      +---------------------------------------------+
                                                             |
                        +------------------------------------+------------------------------------+
                        |                                                                         |
                        v                                                                         v
+-----------------------------------------------+                       +-----------------------------------------------+
|        Module 1: Structured Summarizer        |                       |       Module 2: Key-Element Extractor         |
| - TextRank / LexRank PageRank Graph           |                       | - Case Metadata (Title, Court, Bench, Date)   |
| - Discourse Zone Classifier (IRAC)            |                       | - Decision Outcome Badge (Allowed/Dismissed)  |
| - Legal Salience Priors (Ratio & Holding)     |                       | - Statutory Citations (Acts & Sections)       |
| - Executive Synthesis & Key Takeaways         |                       | - Cross-Referenced Precedent Authorities      |
+-----------------------------------------------+                       +-----------------------------------------------+
                        |                                                                         |
                        +------------------------------------+------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|                                    Module 3: Semantic Precedent Search Engine                                         |
|  - Natural Language Legal Query Input                                                                                 |
|  - Domain Concept Expansion via Legal Thesaurus (Ontology mapping)                                                    |
|  - Hybrid Retrieval: Dense Semantic Vector Cosine Similarity (65%) + Okapi BM25 Keyword Scoring (35%)                 |
|  - Precedent Relevance Explanation & Highlighted Case Excerpts                                                        |
+-----------------------------------------------------------------------------------------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|                                  Module 4: Side-by-Side Judicial Comparison                                           |
|  - Multi-Case Aligned Grid (Facts vs. Facts, Reasoning vs. Reasoning, Ruling vs. Ruling)                              |
|  - Doctrinal Concordance Analysis (Precedent Followed vs. Distinguished on Factual Matrix vs. Conflicting)             |
+-----------------------------------------------------------------------------------------------------------------------+
                                                             |
                                                             v
+-----------------------------------------------------------------------------------------------------------------------+
|                                     Export & Presentation Layer (Web UI & API)                                        |
|  - Interactive IRAC Tabs • PDF/Markdown/JSON Export • Benchmark Dashboard • Legal Disclaimers Banner                  |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Core Modules & NLP Algorithms

### Module 1: Structured IRAC Summarizer (`legal_nlp/summarizer.py`)
- **Discourse Salience Weighting:** Legal judgments have a pronounced non-uniform informational distribution. Sentences in the final $15\%$ (disposition) or opening $15\%$ (factual framing) receive a $+0.35$ to $+0.50$ prior boost. Sentences bearing explicit discourse markers (*"we hold that"*, *"ratio decidendi"*, *"in our considered view"*) receive salience multipliers.
- **Personalized PageRank Kernel:** Builds an adjacency matrix using sublinear TF-IDF weighted cosine similarity between sentences:
  $$\text{Sim}(S_i, S_j) = \frac{\sum_{w} \text{TF}_{w,i} \cdot \text{TF}_{w,j} \cdot \text{IDF}_w^2}{\|V_{S_i}\|_2 \cdot \|V_{S_j}\|_2}$$
  The graph is iteratively resolved to identify central sentences, which are then categorized into IRAC zones.

### Module 2: Key-Element Extractor (`legal_nlp/extractor.py`)
- **Metadata Parser:** Dissects judicial headings to identify parties, court hierarchy (Supreme Court, High Court, House of Lords), decision date, and bench coram.
- **Statutory Provision Recognizer:** Regular expression entity recognizer extracting constitutional provisions (e.g. *Article 21*, *Article 14*, *Article 19(1)(a)*) and statutory codes (*Section 302 IPC*, *Section 66A IT Act*, *Section 3(d) Patents Act*).
- **Outcome Classifier:** Parses concluding sections to determine disposition category (`Allowed`, `Dismissed`, `Acquitted`, `Remanded`, `Partly Allowed`).

### Module 3: Semantic Precedent Search Engine (`legal_nlp/semantic_search.py`)
- **Legal Thesaurus & Concept Expansion:** Maps user queries through a legal domain ontology (e.g., expanding *"surveillance"* to *"personal autonomy"*, *"data protection"*, *"Article 21"*).
- **Hybrid Retrieval Fusion:** Computes a composite relevance score:
  $$\text{Score}_{\text{hybrid}} = \alpha \cdot \text{CosineSim}(V_{\text{query}}, V_{\text{case}}) + (1 - \alpha) \cdot \text{BM25}_{\text{norm}}(Q, D) + \beta \cdot |M_{\text{concepts}}|$$
  where $\alpha = 0.65$ prioritizes semantic alignment and $\beta = 0.08$ rewards explicit conceptual ontology matches.

### Module 4: Case Comparator & Concordance Engine (`legal_nlp/comparator.py`)
- Aligns multiple case briefs side by side across 6 dimensions.
- Computes pairwise reasoning overlap to evaluate whether judgments follow identical precedent doctrines or diverge due to factual matrix distinctions.

---

## 4. Empirical Evaluation & Benchmark Results

The system was evaluated using `evaluate_system.py` against gold-standard reference briefs across representative landmark judgments and standardized benchmark queries:

### 4.1 Summarization Benchmark (ROUGE Metrics)
| Metric | Description | Achieved Score | Target Baseline | Result |
|---|---|---|---|---|
| **ROUGE-1 (F1)** | Unigram overlap (factual terms, entities) | **49.6%** | > 45.0% | **Exceeded** |
| **ROUGE-2 (F1)** | Bigram overlap (legal phrases, doctrines) | **28.4%** | > 20.0% | **Exceeded** |
| **ROUGE-L (F1)** | Longest Common Subsequence (fluency) | **41.2%** | > 35.0% | **Exceeded** |
| **Compression Ratio** | Average reduction from full text to brief | **74.2%** | 65% - 85% | **Optimal** |

### 4.2 Precedent Search Benchmark (Retrieval Accuracy)
| Metric | Description | Achieved Score | Target Baseline | Result |
|---|---|---|---|---|
| **Precision@1 (P@1)** | Authoritative precedent returned as #1 result | **100.0%** | > 80.0% | **Exceeded** |
| **Precision@3 (P@3)** | Relevant candidates in top 3 hits | **66.7%** | > 60.0% | **Exceeded** |
| **Recall@3 (R@3)** | Proportion of all relevant authorities retrieved | **85.0%** | > 75.0% | **Exceeded** |
| **MRR** | Mean Reciprocal Rank ($1 / \text{rank}$) | **1.00** | > 0.70 | **Exceeded** |

*For complete per-case and per-query tables, see [`EVALUATION_REPORT.md`](file:///c:/Users/rishu/OneDrive/Desktop/cse%20CMV%20project/EVALUATION_REPORT.md).*

---

## 5. Curated Landmark Case Repository

LexBrief AI comes pre-loaded with representative landmark cases spanning major areas of law:
1. **Constitutional Law:** *Justice K.S. Puttaswamy v. Union of India (2017)* - Right to Privacy
2. **Constitutional Law:** *Kesavananda Bharati v. State of Kerala (1973)* - Basic Structure Doctrine
3. **Cyber Law & Free Speech:** *Shreya Singhal v. Union of India (2015)* - Section 66A IT Act
4. **Tort & Negligence:** *Donoghue v. Stevenson (1932)* - Duty of Care & Neighbor Principle
5. **Contract Law:** *Hadley v. Baxendale (1854)* - Remoteness of Damages
6. **Criminal Procedure:** *Miranda v. Arizona (1966)* - Custodial Warnings & Rights
7. **Criminal Procedure:** *D.K. Basu v. State of West Bengal (1997)* - Arrest Guidelines & Custodial Violence
8. **Intellectual Property:** *Novartis AG v. Union of India (2013)* - Section 3(d) Patent Evergreening
9. **Environmental Law:** *M.C. Mehta v. Union of India (1987)* - Rule of Absolute Liability
10. **Medical Negligence:** *Jacob Mathew v. State of Punjab (2005)* - Criminal Negligence Standards & Bolam Test

---

## 6. Installation & Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Web browser (Chrome, Edge, Firefox)

### Setup & Run
1. Clone or open the project directory:
   ```powershell
   cd "cse CMV project"
   ```
2. Install dependencies (lightweight, zero heavy GPU requirement):
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the application:
   ```powershell
   python app.py
   ```
4. Access the web dashboard at: **`http://127.0.0.1:5000`**

### Running Unit Tests
```powershell
python -m unittest test_app.py
```

### Running Benchmark Evaluation
```powershell
python evaluate_system.py
```

---

## 7. Project Directory Structure

```
cse CMV project/
├── app.py                      # Flask Application entry point & API routes
├── config.py                   # System hyperparameters and path configuration
├── evaluate_system.py          # Benchmark suite computing ROUGE-1/2/L & P@K/MRR
├── test_app.py                 # Automated unit tests for all NLP modules & routes
├── requirements.txt            # Application dependencies
├── run.bat                     # Windows one-click execution script
├── run.sh                      # Unix/macOS execution script
├── QUICK_START.md              # Viva presentation and demonstration guide
├── EVALUATION_REPORT.md        # Detailed empirical benchmark report
├── README.md                   # Complete documentation
│
├── legal_nlp/                  # Core NLP Architecture
│   ├── __init__.py
│   ├── preprocessor.py         # Citation-safe sentence tokenizer & legal stemmer
│   ├── legal_thesaurus.py      # Legal domain ontology & query expansion engine
│   ├── summarizer.py           # TextRank PageRank summarizer & IRAC brief generator
│   ├── extractor.py            # Key-element extractor (Facts, Arguments, Ruling, Citations)
│   ├── semantic_search.py      # Hybrid BM25 + dense legal concept vector search engine
│   └── comparator.py           # Multi-case comparison matrix & concordance analyzer
│
├── data/                       # Ground-Truth Data & Benchmarks
│   ├── cases.json              # Curated database of 10 landmark judgments
│   ├── benchmark_queries.json  # 10 annotated research queries for evaluation
│   └── evaluation_results.json # Saved quantitative benchmark outputs
│
├── templates/                  # Frontend Jinja2 Views
│   ├── base.html               # Master layout, navigation, ethics modal & disclaimers
│   ├── index.html              # Dashboard with metrics, personas & featured cases
│   ├── summarize.html          # Interactive IRAC summarizer & entity extraction
│   ├── search.html             # Semantic precedent lookup & relevance explanations
│   ├── compare.html            # Side-by-side case comparison & concordance view
│   ├── evaluation.html         # Empirical evaluation dashboard & live re-run
│   └── disclaimer.html         # Scope-of-use & statutory legal disclaimers
│
└── static/                     # Static Frontend Assets
    ├── css/
    │   └── style.css           # Premium legal-tech styling, responsive & print styles
    └── js/
        └── main.js             # Client controller, AJAX handlers, export logic
```

---

## 8. Compliance & Ethical Disclaimers

> [!IMPORTANT]
> **Academic & Research Scope:** LexBrief AI was developed under **Project ID: CSEAIML2026-P13** for academic demonstration and research assistance.
> 
> Outputs produced by this software **do not constitute legal advice** and cannot substitute for representation by a qualified legal advocate. Users must verify all citations and propositions against certified judicial reporters.
