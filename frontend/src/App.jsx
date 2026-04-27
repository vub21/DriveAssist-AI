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

const TrashIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    <path d="M10 11v6M14 11v6" />
    <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
  </svg>
)

const UploadIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
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

function ProgressModal({ filename, log, percent, status, onClose }) {
  const logRef = useRef(null)
  const isDone = status === 'done' || status === 'error'

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [log])

  const title = {
    uploading: 'Uploading Manual…',
    ingesting: 'Training Manual…',
    done: 'Manual Ready',
    error: 'Something Went Wrong',
  }[status] ?? 'Processing…'

  return (
    <div className="modal-overlay">
      <div className="modal">
        <div className="modal-header">
          <p className="modal-title">{title}</p>
          <p className="modal-filename">{filename}</p>
        </div>

        <div className="modal-log" ref={logRef}>
          {status === 'uploading' && (
            <div className="log-item active">
              <span className="log-spinner" />
              Uploading file…
            </div>
          )}
          {log.map((msg, i) => {
            const isLast = i === log.length - 1
            const isActive = isLast && !isDone
            return (
              <div key={i} className={`log-item ${isActive ? 'active' : 'done'}`}>
                {isActive
                  ? <span className="log-spinner" />
                  : <span className="log-check">✓</span>
                }
                {msg}
              </div>
            )
          })}
        </div>

        <div className="modal-progress-row">
          <div className="progress-track">
            <div
              className={`progress-fill ${status === 'error' ? 'error' : ''}`}
              style={{ width: `${percent}%` }}
            />
          </div>
          <span className="progress-pct">{percent}%</span>
        </div>

        {status === 'done' && (
          <p className="modal-success">Select the manual in the sidebar to start chatting.</p>
        )}
        {status === 'error' && (
          <p className="modal-error-msg">Check the backend logs for details.</p>
        )}

        <div className="modal-footer">
          <button className="modal-close-btn" onClick={onClose} disabled={!isDone}>
            {isDone ? 'Close' : 'Processing…'}
          </button>
        </div>
      </div>
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

  // Delete state
  const [confirmDelete, setConfirmDelete] = useState(null) // filename pending delete

  const handleDelete = async (filename) => {
    try {
      await fetch(`${API_BASE}/models/${encodeURIComponent(filename)}`, { method: 'DELETE' })
      setConfirmDelete(null)
      const data = await fetch(`${API_BASE}/models`).then(r => r.json())
      setModels(data.models)
      if (selectedModel === filename) {
        setSelectedModel(data.models[0] ?? null)
        setMessages([])
      }
    } catch {
      setConfirmDelete(null)
    }
  }

  // Upload / ingest state
  const [showModal, setShowModal] = useState(false)
  const [uploadFilename, setUploadFilename] = useState('')
  const [progressLog, setProgressLog] = useState([])
  const [ingestPercent, setIngestPercent] = useState(0)
  const [ingestStatus, setIngestStatus] = useState('idle')
  const fileInputRef = useRef(null)
  const eventSourceRef = useRef(null)

  const refreshModels = () =>
    fetch(`${API_BASE}/models`)
      .then(r => r.json())
      .then(data => setModels(data.models))
      .catch(() => {})

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
    } catch {
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

  const handleFileChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''

    setUploadFilename(file.name)
    setProgressLog([])
    setIngestPercent(0)
    setIngestStatus('uploading')
    setShowModal(true)

    try {
      // 1. Upload the PDF
      const form = new FormData()
      form.append('file', file)
      const uploadRes = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
      if (!uploadRes.ok) throw new Error('Upload failed.')
      const { filename } = await uploadRes.json()

      // 2. Kick off ingestion
      setIngestStatus('ingesting')
      const ingestRes = await fetch(`${API_BASE}/ingest/${encodeURIComponent(filename)}`, { method: 'POST' })
      if (!ingestRes.ok) throw new Error('Failed to start ingestion.')
      const { job_id } = await ingestRes.json()

      // 3. Stream progress via SSE
      const es = new EventSource(`${API_BASE}/ingest/progress/${job_id}`)
      eventSourceRef.current = es

      es.onmessage = (evt) => {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'ping') return

        if (msg.type === 'progress') {
          setProgressLog(prev => [...prev, msg.message])
          setIngestPercent(msg.percent)
        }

        if (msg.type === 'done') {
          es.close()
          eventSourceRef.current = null
          if (msg.success) {
            setIngestStatus('done')
            setIngestPercent(100)
            setProgressLog(prev => [...prev, msg.message])
            refreshModels().then(() => {
              setModels(prev => {
                const match = prev.find(m => m === filename)
                if (match) switchModel(match)
                return prev
              })
            })
            // Refresh and auto-select the new manual
            fetch(`${API_BASE}/models`)
              .then(r => r.json())
              .then(data => {
                setModels(data.models)
                const match = data.models.find(m => m === filename)
                if (match) switchModel(match)
              })
          } else {
            setIngestStatus('error')
            setProgressLog(prev => [...prev, `Error: ${msg.message}`])
          }
        }
      }

      es.onerror = () => {
        es.close()
        eventSourceRef.current = null
        setIngestStatus('error')
        setProgressLog(prev => [...prev, 'Connection to server lost.'])
      }
    } catch (err) {
      setIngestStatus('error')
      setProgressLog(prev => [...prev, `Error: ${err.message}`])
    }
  }

  const closeModal = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    setShowModal(false)
    setIngestStatus('idle')
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
                  <p className="no-models">No manuals loaded yet.</p>
                )}
                {models.map(m => (
                  confirmDelete === m ? (
                    <div key={m} className="model-delete-confirm">
                      <span>Delete {formatModelName(m)}?</span>
                      <div className="delete-confirm-actions">
                        <button className="delete-yes" onClick={() => handleDelete(m)}>Yes</button>
                        <button className="delete-no" onClick={() => setConfirmDelete(null)}>No</button>
                      </div>
                    </div>
                  ) : (
                    <div key={m} className="model-row">
                      <button
                        className={`model-btn ${selectedModel === m ? 'active' : ''}`}
                        onClick={() => switchModel(m)}
                      >
                        <CarIcon />
                        <span>{formatModelName(m)}</span>
                      </button>
                      <button
                        className="model-delete-btn"
                        onClick={() => setConfirmDelete(m)}
                        aria-label={`Delete ${formatModelName(m)}`}
                      >
                        <TrashIcon />
                      </button>
                    </div>
                  )
                ))}
              </div>

              <button className="upload-btn" onClick={() => fileInputRef.current?.click()}>
                <UploadIcon />
                <span>Upload New Manual</span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
            </div>

            <div className="sidebar-footer">
              <div className="badge">GPT-4o &middot; RAG</div>
            </div>
          </>
        )}
      </aside>

      {/* ── Main ── */}
      <main className="main">
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

        {error && <div className="error-banner">{error}</div>}

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

      {/* ── Upload Progress Modal ── */}
      {showModal && (
        <ProgressModal
          filename={uploadFilename}
          log={progressLog}
          percent={ingestPercent}
          status={ingestStatus}
          onClose={closeModal}
        />
      )}
    </div>
  )
}
