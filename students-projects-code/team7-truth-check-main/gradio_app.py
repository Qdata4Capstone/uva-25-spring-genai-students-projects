import gradio as gr
from claims import extract_claims
from evidence import get_wikipedia_evidence
from verify import verify_claim

# Global to store extracted claims across UI steps
extracted_claims = []

def step1_extract(text):
    """Extract factual claims and update dropdown."""
    global extracted_claims
    extracted_claims = extract_claims(text)
    if not extracted_claims:
        return gr.update(choices=[], value=None), "❗ No factual claims found."
    return (
        gr.update(choices=extracted_claims, value=extracted_claims[0]),
        f"✅ {len(extracted_claims)} claim(s) extracted."
    )

def step2_verify(claim):
    """Verify selected claim and return a single-line verdict + evidence."""
    if not claim or "No factual" in claim:
        return "", ""
    evidence = get_wikipedia_evidence(claim)
    result   = verify_claim(claim, evidence)
    
    label  = result.get("label", "Error")
    scores = result.get("scores", {})
    
    # Compute relative confidence if we have scores
    if scores and label in scores:
        total = sum(scores.values())
        rel_conf = (scores[label] / total) if total > 0 else 0.0
        verdict = f"{label} ({rel_conf:.1%} confidence)"
    else:
        verdict = label

    return verdict, evidence

# Build the Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 AI Fact-Checking Assistant")
    gr.Markdown(
        "Enter a paragraph of text. We'll extract factual claims, "
        "retrieve evidence from Wikipedia, and verify each claim."
    )

    text_input = gr.Textbox(lines=6, label="Enter paragraph to check")
    extract_btn = gr.Button("🧠 Extract Claims")

    claims_dropdown = gr.Dropdown(choices=[], label="Select a claim to verify")
    extract_status  = gr.Textbox(label="Status", interactive=False)

    verify_btn     = gr.Button("🔍 Verify Selected Claim")
    verdict_output = gr.Textbox(label="✅ Verdict", interactive=False)
    evidence_output = gr.Textbox(
        label="📄 Wikipedia Evidence", lines=6, interactive=False
    )

    # Wire up interactions
    extract_btn.click(
        fn=step1_extract,
        inputs=text_input,
        outputs=[claims_dropdown, extract_status]
    )
    verify_btn.click(
        fn=step2_verify,
        inputs=claims_dropdown,
        outputs=[verdict_output, evidence_output]
    )

if __name__ == "__main__":
    demo.launch()
