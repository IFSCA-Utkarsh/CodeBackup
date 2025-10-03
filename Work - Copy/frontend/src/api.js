const BASE_URL = import.meta.env.VITE_API_URL;

async function sendChatMessage(message) {
  const res = await fetch(BASE_URL + '/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: message })
  });

  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.text() });
  }
  
  return res.body; // Return ReadableStream for SSE parsing
}

export default { sendChatMessage };
