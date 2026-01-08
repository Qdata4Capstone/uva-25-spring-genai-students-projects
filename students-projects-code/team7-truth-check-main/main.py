# main.py

from claims import extract_claims
from evidence import get_wikipedia_evidence
from verify import verify_claim

input_text = """
I recently visited the Eiffel Tower. It is in Berlin and was built in 1900. 
Shakespeare wrote The Odyssey in the 1600s.
"""

claims = extract_claims(input_text)

for claim in claims:
    print("\n🔎 Claim:", claim)
    evidence = get_wikipedia_evidence(claim)
    print("📄 Evidence:", evidence[:300] + "..." if evidence else "No evidence.")
    label = verify_claim(claim, evidence)
    print("✅ Verdict:", label)
