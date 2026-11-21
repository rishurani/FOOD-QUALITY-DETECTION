"""
Legal Thesaurus & Domain Concept Ontology
Provides legal synonyms, concept clusters, and query expansion dictionaries
for semantic search and precedent retrieval.
"""

LEGAL_ONTOLOGY = {
    "constitutional_law": {
        "concepts": [
            "fundamental rights", "right to privacy", "free speech", "equality before law",
            "basic structure", "judicial review", "ultra vires", "due process", "habeas corpus",
            "article 14", "article 19", "article 21", "article 32", "constitutional amendment",
            "reasonable restrictions", "personal liberty", "sovereignty"
        ],
        "synonyms": {
            "privacy": ["personal autonomy", "data protection", "surveillance", "informational privacy", "right to be let alone", "article 21"],
            "free speech": ["freedom of expression", "censorship", "chilling effect", "sedition", "defamation", "article 19"],
            "equality": ["discrimination", "arbitrariness", "article 14", "equal protection", "classification", "affirmative action"],
            "liberty": ["personal liberty", "detention", "custody", "due process", "procedure established by law"]
        }
    },
    "criminal_law": {
        "concepts": [
            "mens rea", "actus reus", "culpable homicide", "murder", "circumstantial evidence",
            "burden of proof", "beyond reasonable doubt", "anticipatory bail", "custodial violence",
            "confession", "arrest guidelines", "first information report", "police custody",
            "cross examination", "hostile witness"
        ],
        "synonyms": {
            "bail": ["anticipatory bail", "regular bail", "custody", "surety", "section 438", "non-bailable"],
            "arrest": ["detention", "custodial interrogation", "remand", "handcuffing", "guidelines"],
            "evidence": ["circumstantial evidence", "chain of circumstances", "dying declaration", "eyewitness", "forensic proof"],
            "intent": ["mens rea", "knowledge", "premeditation", "motive", "recklessness"]
        }
    },
    "contract_law": {
        "concepts": [
            "offer and acceptance", "consideration", "breach of contract", "damages",
            "remoteness of damage", "frustration of contract", "force majeure", "specific performance",
            "liquidated damages", "unjust enrichment", "void agreement", "privity of contract"
        ],
        "synonyms": {
            "breach": ["default", "non-performance", "repudiation", "failure to perform", "infringement"],
            "damages": ["compensation", "loss of profits", "indemnity", "quantum meruit", "restitution"],
            "frustration": ["impossibility of performance", "force majeure", "supervening event", "destruction of subject matter"],
            "agreement": ["contract", "covenant", "undertaking", "stipulation", "terms"]
        }
    },
    "tort_law": {
        "concepts": [
            "duty of care", "negligence", "standard of care", "breach of duty", "causation",
            "strict liability", "absolute liability", "vicarious liability", "res ipsa loquitur",
            "contributory negligence", "medical negligence", "bolam test", "product liability"
        ],
        "synonyms": {
            "negligence": ["carelessness", "failure to exercise care", "gross negligence", "duty of care", "bolam test"],
            "liability": ["accountability", "responsibility", "tortfeasor", "compensation", "damages"],
            "doctor": ["medical professional", "hospital", "physician", "surgeon", "medical negligence"]
        }
    },
    "intellectual_property": {
        "concepts": [
            "patentability", "inventive step", "prior art", "novelty", "evergreening",
            "patent infringement", "copyright", "fair dealing", "trademark", "passing off",
            "section 3d", "pharmaceutical patents"
        ],
        "synonyms": {
            "patent": ["invention", "monopoly", "prior art", "claims", "patentability", "specification"],
            "infringement": ["violation", "piracy", "counterfeiting", "unauthorized use", "passing off"],
            "evergreening": ["trivial modification", "efficacy", "enhanced therapeutic efficacy", "section 3d"]
        }
    },
    "environmental_law": {
        "concepts": [
            "precautionary principle", "polluter pays principle", "sustainable development",
            "public trust doctrine", "absolute liability", "hazardous industries", "intergenerational equity"
        ],
        "synonyms": {
            "pollution": ["effluent", "emission", "contamination", "toxic discharge", "hazardous waste"],
            "environment": ["ecological balance", "public health", "natural resources", "conservation"]
        }
    }
}

# Discourse markers indicating legal sections in judgments
LEGAL_DISCOURSE_MARKERS = {
    "facts": [
        "the facts of the case", "the brief facts", "the prosecution case is",
        "the case of the petitioner", "procedural history", "the factual matrix",
        "background of the case", "events leading to", "the appellant was charged",
        "the respondent filed a suit", "undisputed facts are"
    ],
    "issues": [
        "the question for consideration", "the point for determination",
        "the issue that arises", "whether the court", "the substantial question of law",
        "the question before us", "we are called upon to determine", "points for decision"
    ],
    "arguments_petitioner": [
        "learned senior counsel for the appellant", "counsel for the petitioner contended",
        "it was submitted on behalf of", "the appellant argues", "the petitioner submits",
        "counsel urged that", "the grievance of the petitioner is"
    ],
    "arguments_respondent": [
        "on the other hand, learned counsel for the respondent", "opposing the appeal, it was contended",
        "counsel for the state submitted", "the respondent contended", "it was strongly urged by the defense",
        "in reply, it was submitted"
    ],
    "reasoning": [
        "we have considered the submissions", "in our considered view", "the law is well settled",
        "it is necessary to examine", "the ratio decidendi of", "having regard to the totality",
        "applying the principle laid down in", "it is evident that", "the court observes",
        "we are unable to accept", "the statutory scheme reveals", "a plain reading of"
    ],
    "ruling": [
        "for the reasons stated above", "in the result, the appeal is allowed",
        "the appeal is dismissed", "the impugned order is set aside", "we hold and declare",
        "accordingly, the writ petition is", "ordered accordingly", "the conviction is upheld",
        "the accused is acquitted", "bail is granted subject to", "no order as to costs"
    ]
}


def expand_query(query: str) -> str:
    """
    Expands a natural language query with legal synonyms and related concepts
    to improve semantic precedent search recall.
    """
    query_lower = query.lower()
    expanded_terms = set(query.split())
    
    for domain, data in LEGAL_ONTOLOGY.items():
        # Check synonyms
        for trigger, synonyms in data.get("synonyms", {}).items():
            if trigger in query_lower:
                expanded_terms.update(synonyms[:3])
        # Check concepts
        for concept in data.get("concepts", []):
            concept_words = concept.split()
            if any(w in query_lower for w in concept_words if len(w) > 3):
                expanded_terms.add(concept)
                
    return " ".join(expanded_terms)
