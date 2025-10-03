import { useState } from "react";

function Login({ onLogin }) {
  const [id, setId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    try {
      const res = await fetch(import.meta.env.VITE_API_URL + "/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          username: id,
          password: password,
        }),
      });

      if (!res.ok) throw new Error("Invalid credentials");
      const data = await res.json();
      localStorage.setItem("token", data.access_token);
      onLogin();
    } catch (err) {
      setError("Login failed: " + err.message);
    }
  }

  return (
    <div className="flex items-center justify-center h-screen">
      <form onSubmit={handleSubmit} className="p-6 bg-white shadow-lg rounded-xl space-y-4">
        <h2 className="text-xl font-bold">Employee Login</h2>
        {error && <p className="text-red-600">{error}</p>}
        <input
          type="text"
          placeholder="Employee ID"
          value={id}
          onChange={e => setId(e.target.value)}
          className="border p-2 w-full"
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          className="border p-2 w-full"
        />
        <button type="submit" className="w-full bg-blue-600 text-white py-2 rounded-md">
          Login
        </button>
      </form>
    </div>
  );
}

export default Login;
