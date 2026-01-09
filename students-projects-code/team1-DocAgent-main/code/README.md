# DocAgent
AI agent Google Docs assistant to help edit research papers and find sources.

---

## Overview ✅
DocAgent is a small command-line AI assistant designed to read, research, and update Google Docs with academic-style edits and citations. It combines Google Docs API access, SerpAPI (for web and Google Scholar searches), and Google Generative AI (Gemini) via LangChain to perform research and produce complete document updates.

## Files 🔧
- `ai.py` — Main implementation. Key components:
  - Environment and API setup (Google Docs API, SerpAPI, Google Generative AI) and `.env` loading.
  - `initialize()` sets up a LangChain agent with tools for web and Google Scholar searches and a conversational memory.
  - `read_google_doc(doc_id)` reads and flattens text from a Google Doc's body.
  - `update_google_doc(doc_id, new_content)` deletes current body content and inserts `new_content` (note: this is a full-replace behavior).
  - `search_google_scholar(query)` and `search_web(query)` format SerpAPI results into a human-readable text block and a `Sources` section.
  - `handle_query(query, doc_id)` builds a task-specific prompt (research vs. edit), invokes the agent, and handles retries.
  - `main()` provides a simple interactive loop to accept queries, preview proposed document updates, and optionally apply changes.
- `dependencies.txt` — Python dependencies required to run the assistant (LangChain, Google client libraries, SerpAPI wrapper, python-dotenv, etc.).

## Quick Setup & Usage ⚙️
1. Create a `.env` file or export these environment variables:
   - `GOOGLE_APPLICATION_CREDENTIALS` → path to service account JSON (must have Docs scope)
   - `GOOGLE_API_KEY` → API key for Google Generative/other Google APIs
   - `SERPAPI_API_KEY` → API key for SerpAPI
   - (optional) `GOOGLE_DOC_ID` → document to connect to for quick start
2. Install dependencies:
   ```bash
   pip install -r dependencies.txt
   ```
3. Run the assistant:
   ```bash
   python ai.py
   ```
   - Choose a citation style when prompted (apa, mla, harvard, chicago, vancouver).
   - Enter queries (research or edit tasks). The agent will return a complete document text; you can preview and choose whether to apply it to the Google Doc.

## Notes & Caveats ⚠️
- `read_google_doc` flattens document text and ignores structural formatting (styles, headings, lists). The assistant treats Google Docs content as plain text.
- `update_google_doc` currently deletes and reinserts the full document body. This can remove non-textual content or metadata and does not preserve granular formatting or structured elements.
- Citations are generated from SerpAPI results and formatted according to the selected `cite_style`, but the output may need manual verification for strict academic requirements.
- The `ChatGoogleGenerativeAI` model used (`gemini-2.0-flash`) requires access and proper billing setup for the Google Generative API.
- Keep service account credentials private — do not commit them to source control.

## Suggested Improvements 💡
- Preserve and reapply Google Doc structural elements (headings, lists, formatting) instead of plain-text replace.
- Add robust logging, unit tests, and rate-limit/backoff handling for external APIs.
- Add an optional dry-run mode to compare diffs instead of full replacement.
- Improve citation parsing and validation for different styles (APA/MLA/Chicago, etc.).

---

**Feel free to ask** if you want me to expand any section (setup, testing, or architecture diagrams) or to add examples and tests for `ai.py`.
