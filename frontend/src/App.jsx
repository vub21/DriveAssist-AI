import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'

const CarIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 17H3a2 2 0 0 1-2-2V9l2.5-5h13L19 9v6a2 2 0 0 1-2 2h-2" />
    <circle cx="7.5" cy="17" r="2.5" />
    <circle cx="16.5" cy="17" r="2.5" />
    <path d="M5 9h14" />
  </svg>
)

const BotIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" />
    <path d="M12 2a3 3 0 0 1 3 3v6H9V5a3 3 0 0 1 3-3z" />
    <circle cx="9" cy="16" r="1" fill="currentColor" stroke="none" />
    <circle cx="15" cy="16" r="1" fill="currentColor" stroke="none" />
    <path d="M9 20h6" />
  </svg>
)

const SendIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
)

const SUGGESTIONS = [
  'How do I check the engine oil level?',
  'What does the tire pressure warning light mean?',
  'How do I reset the maintenance reminder?',
  'What type of fuel does this vehicle require?',
]

function formatModelName(filename) {
  if (!filename) return 'All Manuals'
  return filename
    .replace(/\.pdf$/i, '')
    .replace(/_/g, ' ')
    .replace(/^\d+\s*/, '')
    .trim()
}

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span /><span /><span />
    </div>
  )
}

function SourceChips({ sources }) {
  if (!sources || sources.length === 0) return null
  return (
    <div className="sources">
      <p className="sources-label">Sources</p>
      <div className="source-chips">
        {sources.map(s => (
          <span key={s.index} className="source-chip">
            [{s.index}] {formatModelName(s.source)} &middot; p.{s.page}
          </span>
        ))}
      </div>
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      {!isUser && (
        <div className="avatar bot-avatar">
          <BotIcon />
        </div>
      )}
      <div className="message-body">
        <div className="bubble">{msg.content}</div>
        {!isUser && <SourceChips sources={msg.sources} />}
      </div>
      {isUser && <div className="avatar user-avatar">You</div>}
    </div>
  )
}

export default function App() {
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then(r => r.json())
      .then(data => {
        setModels(data.models)
        if (data.models.length > 0) setSelectedModel(data.models[0])
      })
      .catch(() => setError('Could not connect to backend. Make sure api.py is running.'))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const autoResize = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  const sendMessage = async (text) => {
    const query = (text ?? input).trim()
    if (!query || loading) return

    setMessages(prev => [...prev, { role: 'user', content: query }])
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, model: selectedModel }),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
      }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please check that the backend server is running.',
        sources: [],
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const switchModel = (m) => {
    setSelectedModel(m)
    setMessages([])
    setError(null)
  }

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <div className="logo">
            <CarIcon />
            <span>DriveAssist AI</span>
          </div>
          <button className="sidebar-toggle" onClick={() => setSidebarOpen(o => !o)} aria-label="Toggle sidebar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        </div>

        {sidebarOpen && (
          <>
            <div className="sidebar-section">
              <p className="section-label">Vehicle Manual</p>
              <div className="model-list">
                {models.length === 0 && (
                  <p className="no-models">No manuals loaded. Run ingest.py first.</p>
                )}
                {models.map(m => (
                  <button
                    key={m}
                    className={`model-btn ${selectedModel === m ? 'active' : ''}`}
                    onClick={() => switchModel(m)}
                  >
                    <CarIcon />
                    <span>{formatModelName(m)}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="sidebar-footer">
              <div className="badge">GPT-4o &middot; RAG</div>
            </div>
          </>
        )}
      </aside>

      {/* ── Main ── */}
      <main className="main">
        {/* Header */}
        <header className="chat-header">
          <div className="header-info">
            <h1>Vehicle Assistant</h1>
            <p className="header-sub">
              {selectedModel ? formatModelName(selectedModel) : 'All Manuals'}
            </p>
          </div>
          {messages.length > 0 && (
            <button className="clear-btn" onClick={() => setMessages([])}>
              Clear chat
            </button>
          )}
        </header>

        {/* Error banner */}
        {error && (
          <div className="error-banner">{error}</div>
        )}

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <div className="empty-icon"><CarIcon /></div>
              <h2>Ask anything about your vehicle</h2>
              <p>Select a manual in the sidebar, then ask a question below.</p>
              <div className="suggestions">
                {SUGGESTIONS.map(s => (
                  <button key={s} className="suggestion-btn" onClick={() => sendMessage(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => <Message key={i} msg={msg} />)}

          {loading && (
            <div className="message assistant">
              <div className="avatar bot-avatar"><BotIcon /></div>
              <div className="message-body">
                <div className="bubble"><TypingIndicator /></div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="input-row">
          <div className="input-box">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => { setInput(e.target.value); autoResize() }}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your vehicle…"
              rows={1}
              disabled={loading}
            />
            <button
              className="send-btn"
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              aria-label="Send"
            >
              <SendIcon />
            </button>
          </div>
          <p className="input-hint">Press Enter to send &middot; Shift+Enter for new line</p>
        </div>
      </main>
    </div>
  )
}
