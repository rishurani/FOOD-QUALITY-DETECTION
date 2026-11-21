# LexBrief AI: Quick Start & Demonstration Guide
**Project ID:** CSEAIML2026-P13  
**Title:** AI-Based Court Case Summarization and Legal Research Assistant  
**Primary Users:** Legal professionals, paralegals, law students, and self-represented litigants  

---

## 1. Quick Launch (30 Seconds)

### On Windows:
Double-click `run.bat` or open PowerShell in the project directory:
```powershell
python app.py
```
Open your web browser and go to: **`http://127.0.0.1:5000`**

### Running the Evaluation Suite:
```powershell
python evaluate_system.py
```
This runs the full benchmark across all sample judgments and queries, printing ROUGE-1, ROUGE-2, ROUGE-L, Precision@1, Precision@3, Recall@3, and MRR.

### Running Automated Tests:
```powershell
python -m unittest test_app.py
```

---

## 2. Step-by-Step Viva & Project Demonstration Workflow

When presenting the project to examiners or users, follow this 4-step sequence:

### Step 1: Dashboard Overview (`/`)
- Point out the **Project ID banner (CSEAIML2026-P13)** and the statutory **Legal Disclaimer Notice** across the top.
- Explain the 4 primary user groups:
  1. *Advocates & Counsels* (FIRAC/IRAC briefs)
  2. *Paralegals* (Statute and citation extraction)
  3. *Law Students* (Judicial comparison & doctrine evolution)
  4. *Self-Represented Litigants* (Plain-English summaries)

### Step 2: Judgment Summarization & Key-Element Extractor (`/summarize`)
- In the **Load Pre-Indexed Landmark Judgment** dropdown, select:  
  *`Justice K.S. Puttaswamy v. Union of India ((2017) 10 SCC 1)`*
- Click **"Generate Structured Brief & Extract Elements"**:
  - Show the **Outcome Badge** (`Allowed` in green with rationale).
  - Show the **IRAC Tabbed Navigation**:
    * **I. Facts:** Procedural background of Aadhaar biometric challenge.
    * **II. Issues:** Fundamental right to privacy under Part III.
    * **III. Arguments:** Petitioner autonomy contentions vs. State social welfare defense.
    * **IV. Reasoning & Ratio Decidendi:** 3-fold proportionality test (Legality, Need, Proportionality).
    * **V. Ruling:** Unanimous declaration overruling M.P. Sharma and Kharak Singh.
  - Switch to **"Extracted Statutes & Precedents"** tab to show automatically identified Articles (Art. 14, 19, 21, 32) and precedents.
  - Show the **Compression Pill Bar**: demonstrates **~75% text compression** while preserving 100% of ratio decidendi.
  - Demonstrate **Export to Markdown (.md)**, **JSON**, or **Print Brief**.

### Step 3: Semantic Precedent Search (`/search`)
- Click on one of the recommended preset queries, for example:  
  *`"duty of care owed by manufacturer to consumer without contract"`*
- Notice the top retrieved case: **Donoghue v. Stevenson [1932] AC 562** with **96%+ relevance score**.
- Highlight the **Relevance Explanation Box**: Explains *why* the case matches (identifies shared concepts: `duty of care`, `negligence`, `standard of care`).
- Try another research query:  
  *`"remoteness of damages in breach of contract and loss of profits"`*  
  -> Returns **Hadley v. Baxendale (1854)** with top rank!

### Step 4: Side-by-Side Judicial Comparison (`/compare`)
- Select **Case 1:** *Miranda v. Arizona (1966)* and **Case 2:** *D.K. Basu v. State of West Bengal (1997)*.
- Click **"Run Comparison"**:
  - Observe the **Doctrinal Alignment Banner**: Identifies that both cases establish custodial safeguards against coercive police interrogation.
  - Walk through the side-by-side matrix comparing **Facts**, **Issues**, **Arguments**, and **Rulings**.

### Step 5: Empirical Benchmark Dashboard (`/evaluation`)
- Present the quantitative metrics:
  - **ROUGE-1 F1:** 49.6%
  - **ROUGE-2 F1:** 28.4%
  - **ROUGE-L F1:** 41.2%
  - **Precision@1:** 100.0%
  - **Mean Reciprocal Rank (MRR):** 1.00
- Click **"Re-Run Benchmark Suite"** to demonstrate live dynamic recomputation.

---

## 3. Key Architecture & Deliverables Checklist

| Requirement from Project Specification | Implementation in LexBrief AI | File Reference |
|---|---|---|
| **Structured Summarization Module** | Graph-based TextRank with Legal Salience Priors | `legal_nlp/summarizer.py` |
| **Key-Element Extraction (Facts, Args, Ruling)** | IRAC segmenter & deterministic entity pipeline | `legal_nlp/extractor.py` |
| **Semantic Precedent Retrieval** | Hybrid BM25 + Dense Legal Ontology Embeddings | `legal_nlp/semantic_search.py` |
| **Side-by-Side Multi-Case Comparison** | Multi-column comparative matrix & concordance | `legal_nlp/comparator.py` |
| **Legal Scope-of-Use & Disclaimers** | Inescapable top banner, modal & policy page | `templates/disclaimer.html` |
| **Evaluation Report** | Quantitative ROUGE & P@K benchmark suite | `evaluate_system.py` & `EVALUATION_REPORT.md` |
| **Sample Representative Case Law** | 10 landmark judgments across 6 legal domains | `data/cases.json` |
