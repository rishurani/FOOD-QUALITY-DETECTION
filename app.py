"""
LexBrief AI - Legal Case Summarization & Precedent Research Platform
Flask Application Entry Point and API Router
Project ID: CSEAIML2026-P13
"""
import json
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, make_response, send_file

from config import Config
from legal_nlp.preprocessor import LegalPreprocessor
from legal_nlp.summarizer import LegalSummarizer
from legal_nlp.extractor import LegalElementExtractor
from legal_nlp.semantic_search import SemanticSearchEngine
from legal_nlp.comparator import LegalCaseComparator
from evaluate_system import run_evaluation

# Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# Ensure directories exist
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.UPLOADS_DIR, exist_ok=True)

# Global NLP Singletons
summarizer = LegalSummarizer()
extractor = LegalElementExtractor()
search_engine = SemanticSearchEngine(
    k1=Config.BM25_K1,
    b=Config.BM25_B,
    semantic_weight=Config.SEMANTIC_WEIGHT
)

# Load Cases Database
CASES_DB = []
CASES_MAP = {}

def load_database():
    """Loads cases from data/cases.json and indexes them in the semantic search engine."""
    global CASES_DB, CASES_MAP
    if Config.CASES_JSON.exists():
        with open(Config.CASES_JSON, 'r', encoding='utf-8') as f:
            CASES_DB = json.load(f)
            CASES_MAP = {c["id"]: c for c in CASES_DB}
            search_engine.index_cases(CASES_DB)
            print(f"[LexBrief AI] Successfully indexed {len(CASES_DB)} landmark cases.")

load_database()


# -----------------------------------------------------------------------------
# Frontend View Routes
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    """Dashboard & Home Overview."""
    domains = sorted(list(set(c.get("domain", "General") for c in CASES_DB)))
    total_words = sum(len(c.get("text", "").split()) for c in CASES_DB)
    return render_template(
        'index.html',
        cases=CASES_DB,
        domains=domains,
        total_cases=len(CASES_DB),
        total_words=total_words,
        project_id=Config.PROJECT_ID,
        project_name=Config.PROJECT_NAME
    )


@app.route('/summarize')
def summarize_page():
    """Legal Case Summarization & Key-Element Extraction view."""
    case_id = request.args.get('case_id')
    selected_case = CASES_MAP.get(case_id) if case_id else None
    return render_template(
        'summarize.html',
        cases=CASES_DB,
        selected_case=selected_case,
        project_id=Config.PROJECT_ID
    )


@app.route('/search')
def search_page():
    """Semantic Precedent Search view."""
    query = request.args.get('q', '')
    domain = request.args.get('domain', 'all')
    domains = sorted(list(set(c.get("domain", "General") for c in CASES_DB)))
    
    results = []
    if query:
        results = search_engine.search(query, domain_filter=domain, top_k=6)

    return render_template(
        'search.html',
        query=query,
        domain=domain,
        domains=domains,
        results=results,
        project_id=Config.PROJECT_ID
    )


@app.route('/compare')
def compare_page():
    """Side-by-Side Multi-Case Comparison view."""
    case_ids = request.args.getlist('case_id')
    if not case_ids and len(CASES_DB) >= 2:
        # Default compare first 2 cases
        case_ids = [CASES_DB[0]["id"], CASES_DB[1]["id"]]

    selected_cases = [CASES_MAP[cid] for cid in case_ids if cid in CASES_MAP]
    comparison_data = LegalCaseComparator.compare_cases(selected_cases) if len(selected_cases) >= 2 else None

    return render_template(
        'compare.html',
        cases=CASES_DB,
        selected_ids=case_ids,
        comparison=comparison_data,
        project_id=Config.PROJECT_ID
    )


@app.route('/evaluation')
def evaluation_page():
    """Empirical Evaluation & Benchmarking Dashboard."""
    results = None
    if Config.EVALUATION_JSON.exists():
        try:
            with open(Config.EVALUATION_JSON, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except Exception:
            pass

    if not results:
        results = run_evaluation()

    return render_template(
        'evaluation.html',
        eval_data=results,
        project_id=Config.PROJECT_ID
    )


@app.route('/disclaimer')
def disclaimer_page():
    """Full legal disclaimer and scope-of-use policy."""
    return render_template('disclaimer.html', project_id=Config.PROJECT_ID)


# -----------------------------------------------------------------------------
# RESTful API Endpoints
# -----------------------------------------------------------------------------

@app.route('/api/cases', methods=['GET'])
def api_list_cases():
    """Returns directory of all indexed landmark judgments."""
    summary_list = [
        {
            "id": c["id"],
            "title": c["title"],
            "citation": c["citation"],
            "court": c["court"],
            "date": c["date"],
            "domain": c["domain"],
            "outcome": c.get("outcome", {})
        }
        for c in CASES_DB
    ]
    return jsonify({"success": True, "cases": summary_list})


@app.route('/api/cases/<case_id>', methods=['GET'])
def api_get_case(case_id):
    """Returns complete case record and metadata by ID."""
    case = CASES_MAP.get(case_id)
    if not case:
        return jsonify({"success": False, "error": f"Case ID '{case_id}' not found"}), 404
    return jsonify({"success": True, "case": case})


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """
    Summarizes court judgment text or an existing case ID.
    Extracts metadata, facts, issues, arguments, reasoning, ruling, and statutes.
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    case_id = data.get('case_id')
    mode = data.get('mode', 'structured')
    focus = data.get('focus', 'all')
    target_sentences = int(data.get('target_sentences', 8))

    if case_id and case_id in CASES_MAP:
        case = CASES_MAP[case_id]
        text = case["text"]
        title = case["title"]
        citation = case["citation"]
        court = case["court"]
        domain = case["domain"]
        pre_extracted = {
            "metadata": {
                "title": title,
                "citation": citation,
                "court": court,
                "bench": case.get("bench", []),
                "date": case.get("date", "")
            },
            "domain": domain,
            "statutes_cited": case.get("statutes_cited", []),
            "precedents_cited": case.get("precedents_cited", []),
            "outcome": case.get("outcome", {})
        }
    else:
        if not text.strip():
            return jsonify({"success": False, "error": "No judgment text provided."}), 400
        pre_extracted = extractor.extract_all_elements(text)

    # Generate brief using TextRank + Legal Salience weighting
    brief = summarizer.generate_brief(
        text,
        target_sentences=target_sentences,
        mode=mode,
        focus=focus
    )

    response_payload = {
        "success": True,
        "brief": brief,
        "extracted_elements": pre_extracted,
        "disclaimer": "This summary and extracted briefs are generated by LexBrief AI for legal research assistance only. Not legal advice."
    }
    return jsonify(response_payload)


@app.route('/api/search', methods=['POST'])
def api_search():
    """Performs semantic and hybrid precedent retrieval."""
    data = request.get_json(silent=True) or {}
    query = data.get('query', '')
    domain = data.get('domain', 'all')
    outcome = data.get('outcome', 'all')
    top_k = int(data.get('top_k', 5))

    if not query.strip():
        return jsonify({"success": False, "error": "Search query cannot be empty."}), 400

    results = search_engine.search(query, domain_filter=domain, outcome_filter=outcome, top_k=top_k)
    return jsonify({
        "success": True,
        "query": query,
        "total_results": len(results),
        "results": results
    })


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Compares multiple cases side by side."""
    data = request.get_json(silent=True) or {}
    case_ids = data.get('case_ids', [])

    if len(case_ids) < 2:
        return jsonify({"success": False, "error": "Please provide at least 2 case IDs to compare."}), 400

    selected_cases = [CASES_MAP[cid] for cid in case_ids if cid in CASES_MAP]
    if len(selected_cases) < 2:
        return jsonify({"success": False, "error": "Could not locate provided case IDs."}), 404

    result = LegalCaseComparator.compare_cases(selected_cases)
    return jsonify({"success": True, "comparison": result})


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Triggers the evaluation and benchmark suite."""
    results = run_evaluation()
    return jsonify({"success": True, "evaluation": results})


@app.route('/api/export', methods=['POST'])
def api_export():
    """Exports structured brief in Markdown, JSON, or text format."""
    data = request.get_json(silent=True) or {}
    fmt = data.get('format', 'markdown').lower()
    title = data.get('title', 'Court_Case_Brief')
    content = data.get('content', '')

    safe_title = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in title)

    if fmt == 'json':
        resp = make_response(json.dumps(content, indent=2))
        resp.headers["Content-Disposition"] = f"attachment; filename={safe_title}.json"
        resp.headers["Content-Type"] = "application/json"
        return resp
    else:
        # Default markdown
        resp = make_response(content)
        resp.headers["Content-Disposition"] = f"attachment; filename={safe_title}.md"
        resp.headers["Content-Type"] = "text/markdown; charset=utf-8"
        return resp


# -----------------------------------------------------------------------------
# Main Application Launcher
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    print("=" * 70)
    print(" LexBrief AI: Court Case Summarization & Legal Research Assistant ")
    print(f" Project ID: {Config.PROJECT_ID}")
    print(f" Server active on: http://127.0.0.1:{port}")
    print("=" * 70)
    app.run(host='127.0.0.1', port=port, debug=debug)
