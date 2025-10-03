const BASE_URL = import.meta.env.VITE_API_URL;

function getAuthHeaders() {
  const token = localStorage.getItem("token");
  return token ? { Authorization: "Bearer " + token } : {};
}

// Login
async function login(id, password) {
  const res = await fetch(BASE_URL + "/login", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      username: id,
      password: password,
    }),
  });

  if (!res.ok) {
    throw new Error(await res.text());
  }

  return res.json(); // contains { access_token, token_type }
}

// Chat
async function sendChatMessage(message) {
  const res = await fetch(BASE_URL + "/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
    },
    body: JSON.stringify({ question: message }),
  });

  if (!res.ok) {
    return Promise.reject({ status: res.status, data: await res.text() });
  }

  return res.body; // ReadableStream
}

export default { login, sendChatMessage };
