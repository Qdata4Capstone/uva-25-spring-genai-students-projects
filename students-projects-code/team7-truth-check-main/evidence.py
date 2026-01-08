# evidence.py
from models import nlp
import wikipedia

def extract_best_query(claim: str) -> str:
    """
    Pick the strongest search term:
    1) Named entity of type PERSON, ORG, GPE, LOC, EVENT
    2) Otherwise fallback to full claim.
    """
    doc = nlp(claim)
    ents = [
        ent.text
        for ent in doc.ents
        if ent.label_ in {"PERSON","ORG","GPE","LOC","EVENT"}
    ]
    return ents[0] if ents else claim

def get_wikipedia_evidence(claim: str) -> str:
    """
    1. Build a short query via extract_best_query.
    2. Search Wikipedia (up to 3 results).
    3. Return first 1000 chars of the first page that contains the query.
    4. Else return "No evidence found".
    """
    query = extract_best_query(claim)
    try:
        for title in wikipedia.search(query)[:3]:
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content
            if query.lower() in text.lower():
                return text[:1000]
    except Exception:
        # disambiguation, network errors, etc.
        pass

    return "No evidence found"
