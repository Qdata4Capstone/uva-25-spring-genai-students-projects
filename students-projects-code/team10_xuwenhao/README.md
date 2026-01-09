# Team 10 - AI Blog Forum + Generator

This project is a React single-page app that lets users write and publish blog posts, with an AI-assisted post generator. It includes a lightweight PDF extraction service and an OpenAI-backed text generator.

## Folder map
- App.js: Route definitions and auth-guarded pages.
- components/: UI pages (login, forum, create post, AI generator, post detail).
- api/openai.js: OpenAI client wrapper (chat completions).
- prompts/systemPrompt.js: Base system prompt for blog generation.
- server/server.js: Express service that extracts text from uploaded PDFs.
- publics/: Static assets for the React app.

## How it works
1. Users sign up/login (stored in `localStorage`).
2. Posts are created/edited in a Markdown editor.
3. The AI generator builds a prompt from topic + keywords + PDF excerpt.
4. `api/openai.js` calls OpenAI to generate a blog draft.
5. Posts are stored locally and shown in the forum.

## Setup
This is a React app with a small Node service. You need Node.js 18+.

### Install dependencies
From this folder:

```
npm install
```

### Important library: OpenAI SDK
The AI generator calls OpenAI. Install the SDK (if not already in your `package.json`):

```
npm install openai
```

Then add your API key. The current code uses a placeholder in `api/openai.js`:

```
const OPENAI_API_KEY = 'your-key'
```

Replace it with a real key or load from environment variables via a `.env` file.

## Run the app
Start the React dev server:

```
npm start
```

The app uses client-side PDF parsing via PDF.js from a CDN, so the backend is optional for basic usage.

## Run the PDF extraction service (optional)
If you want server-side PDF parsing, run:

```
node server/server.js
```

This starts an Express server on port 3001 with `/api/extract`.

## Notes and gotchas
- There is no `package.json` in this folder; you may need to scaffold one or move these files into an existing React project.
- API keys should never be hardcoded for production use.
- Posts and users live in `localStorage`, so they reset per browser or when storage is cleared.

