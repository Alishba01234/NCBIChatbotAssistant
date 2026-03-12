import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const BASE_URL = "http://127.0.0.1:8000";

// ── API helpers ─────────────────────────────
async function askQuestion(question) {
  const res = await fetch(`${BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return res.json();
}

async function uploadFiles(files) {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  const res = await fetch(`${BASE_URL}/upload`, { method: "POST", body: formData });
  return res.json();
}

async function indexDocuments() {
  const res = await fetch(`${BASE_URL}/index`, { method: "POST" });
  return res.json();
}

async function fetchChatHistory() {
  const res = await fetch(`${BASE_URL}/chat-history`);
  return res.json();
}

// ── Utilities ───────────────────────────────
function formatTime(iso) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDateLabel(iso) {
  const d = new Date(iso);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);

  if (d.toDateString() === today.toDateString()) return "Today";
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday";

  return d.toLocaleDateString([], { day: "numeric", month: "long", year: "numeric" });
}

function groupByDate(messages) {
  const groups = [];
  let lastLabel = null;

  messages.forEach((msg) => {
    const label = formatDateLabel(msg.timestamp);
    if (label !== lastLabel) {
      groups.push({ type: "date", label });
      lastLabel = label;
    }
    groups.push(msg);
  });

  return groups;
}

// ── Components ──────────────────────────────
function TypingIndicator() {
  return (
    <div className="bubble-row bot">
      <div className="avatar">🧬</div>
      <div className="bubble bot">
        <div className="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  );
}

function DateSeparator({ label }) {
  return (
    <div className="date-sep">
      <span className="date-pill">{label}</span>
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";

  return (
    <div className={`bubble-row ${isUser ? "user" : "bot"}`}>
      {!isUser && <div className="avatar">🧬</div>}

      <div className={`bubble ${isUser ? "user" : "bot"}`}>
        <p className="msg-text">{msg.content}</p>
        <div className="msg-meta">
          <span className="msg-time">{formatTime(msg.timestamp)}</span>
          {isUser && <span className="tick">✓✓</span>}
        </div>
      </div>

      {isUser && <div className="user-avatar">👤</div>}
    </div>
  );
}

// ── Main App ─────────────────────────────────
export default function App() {

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [showSidebar, setShowSidebar] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [files, setFiles] = useState([]);

  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Load chat history
  useEffect(() => {
    (async () => {
      try {
        const data = await fetchChatHistory();
        if (data.history?.length) {
          setMessages(data.history);
        } else {
          setMessages([{
            id: "welcome",
            role: "bot",
            content:
              "👋 Hello! I'm your NCBI Medical Assistant. Ask me anything about diseases, treatments, or clinical research.",
            timestamp: new Date().toISOString(),
          }]);
        }
      } finally {
        setHistoryLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;

    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  }, [input]);

  const sendMessage = async () => {

    const text = input.trim();
    if (!text || loading) return;

    const userMsg = {
      id: Date.now(),
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await askQuestion(text);

      const botMsg = {
        id: Date.now(),
        role: "bot",
        content: res.answer || "No answer found",
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "bot",
          content: "⚠️ Backend connection error.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const grouped = groupByDate(messages);

  return (
    <div className="root">

      {/* Sidebar */}
      <aside className={`sidebar ${showSidebar ? "show" : ""}`}>
        <div className="sidebar-header">
          <span>⚕️ NCBI Assistant</span>
          <button onClick={() => setShowSidebar(false)}>✕</button>
        </div>

        <div className="sidebar-section">
          <p className="sidebar-label">UPLOAD DOCUMENTS</p>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => setFiles([...e.target.files])}
          />

          <button onClick={() => uploadFiles(files)}>Upload</button>
          <button onClick={indexDocuments}>Re-Index</button>

          <p>{uploadStatus}</p>
        </div>
      </aside>

      {/* Chat */}
      <div className="chat-panel">

        <header className="topbar">
          <button onClick={() => setShowSidebar(true)}>☰</button>
          <div className="topbar-title">NCBI Medical Assistant</div>
        </header>

        <div className="messages-area">

          {historyLoading ? (
            <p className="loading">Loading...</p>
          ) : (
            grouped.map((item, i) =>
              item.type === "date" ? (
                <DateSeparator key={i} label={item.label} />
              ) : (
                <Message key={item.id} msg={item} />
              )
            )
          )}

          {loading && <TypingIndicator />}
          <div ref={bottomRef}></div>
        </div>

        <div className="input-bar">

          <textarea
            ref={textareaRef}
            value={input}
            placeholder="Ask a medical question..."
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          <button onClick={sendMessage}>Send</button>

        </div>
      </div>
    </div>
  );
}
