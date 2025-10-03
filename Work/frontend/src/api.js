const BASE_URL = import.meta.env.VITE_API_URL;
const API_KEY = import.meta.env.VITE_API_KEY;

async function sendChatMessage(message) {
  const res = await fetch(BASE_URL + '/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY ? { 'X-API-Key': API_KEY } : {})
    },
    body: JSON.stringify({ question: message })
  });

  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.text() });
  }

  // Return ReadableStream for SSE parsing
  return res.body;
}

export default { sendChatMessage };
