import axios from 'axios';

const OPENAI_API_KEY = 'your-key'; // ⚠️ replace with your key or load from env vars

export async function generateBlogPost(finalPrompt) {
  const response = await axios.post('https://api.openai.com/v1/chat/completions', {
    model: "gpt-4.1",
    messages: [
      { role: "user", content: finalPrompt }
    ],
    temperature: 0.7
  }, {
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    }
  });

  return response.data.choices[0].message.content.trim();
}
