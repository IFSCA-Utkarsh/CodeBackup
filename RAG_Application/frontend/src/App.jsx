import { useState } from "react";
import Chatbot from "@/components/Chatbot";
import api from "@/api";

function Login({ onLogin }) {
  const [id, setId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    try {
      const data = await api.login(id, password);
      localStorage.setItem("token", data.access_token);
      onLogin();
    } catch (err) {
      setError("Login failed. Check your ID/password.");
    }
  }

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <form
        onSubmit={handleSubmit}
        className="p-6 bg-white shadow-lg rounded-xl space-y-4 w-80"
      >
        <h2 className="text-xl font-bold text-center">Employee Login</h2>
        {error && <p className="text-red-600 text-sm">{error}</p>}
        <input
          type="text"
          placeholder="Employee ID"
          value={id}
          onChange={(e) => setId(e.target.value)}
          className="border p-2 w-full rounded"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="border p-2 w-full rounded"
        />
        <button
          type="submit"
          className="w-full bg-blue-600 text-white py-2 rounded-md"
        >
          Login
        </button>
      </form>
    </div>
  );
}

function App() {
  const [loggedIn, setLoggedIn] = useState(!!localStorage.getItem("token"));

  return loggedIn ? (
    <div className="flex flex-col min-h-full w-full max-w-3xl mx-auto px-4">
      <header className="sticky top-0 bg-white z-20 shadow-sm">
        <h1 className="font-bold text-xl py-2">IFSCA IntelliChat</h1>
      </header>
      <Chatbot />
    </div>
  ) : (
    <Login onLogin={() => setLoggedIn(true)} />
  );
}

export default App;
