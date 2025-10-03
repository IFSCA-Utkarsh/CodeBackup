import { useState, useEffect, useRef } from "react";

export default function Chatbot({ token }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const ws = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    ws.current = new WebSocket(`ws://localhost:8000/ws/${token}`);

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      simulateStreaming(data.answer, () => {
        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: data.answer,
            timestamp: new Date(),
            sources: data.sources || [],
          },
        ]);
        setTyping(false);
      });
    };

    ws.current.onclose = () => {
      alert("Connection closed. Please reload.");
    };

    return () => ws.current?.close();
  }, [token]);

  const simulateStreaming = (text, callback) => {
    setStreamedAnswer("");
    setTyping(true);
    let index = 0;

    const interval = setInterval(() => {
      if (index < text.length) {
        setStreamedAnswer((prev) => prev + text[index]);
        index++;
      } else {
        clearInterval(interval);
        callback();
      }
    }, 20);
  };

  const sendMessage = () => {
    if (!input.trim()) return;
    const userMsg = {
      sender: "user",
      text: input,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    ws.current.send(input);
    setInput("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (date) =>
    new Date(date).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamedAnswer]);

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
            <div className="max-w-sm">
              <div
                className={`px-4 py-2 rounded-lg shadow ${
                  msg.sender === "user" ? "bg-blue-500 text-white" : "bg-white text-gray-800"
                }`}
              >
                {msg.text}
              </div>
              <div className="text-xs text-gray-500 mt-1">{formatTime(msg.timestamp)}</div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2 bg-gray-100 text-sm border-l-4 border-blue-400 pl-3 py-2">
                  <div className="font-semibold mb-1">Sources:</div>
                  {msg.sources.map((src, i) => (
                    <div key={i} className="mb-1">
                      <span className="font-medium">{src.source}:</span>{" "}
                      {src.content.slice(0, 150)}...
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {typing && (
          <div className="flex justify-start">
            <div className="bg-white text-gray-800 px-4 py-2 rounded-lg shadow max-w-sm">
              {streamedAnswer}
              <span className="animate-pulse">|</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <div className="p-4 border-t bg-white">
        <textarea
          rows={1}
          className="w-full border rounded-lg p-2 focus:outline-none resize-none"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
        />
        <button
          className="mt-2 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          onClick={sendMessage}
          disabled={typing || input.trim() === ""}
        >
          Send
        </button>
      </div>
    </div>
  );
}
