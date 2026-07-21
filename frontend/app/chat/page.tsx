'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { Send, Brain, Menu, MessageSquarePlus, Mic, MicOff, Volume2, VolumeX } from 'lucide-react'
import { clsx } from 'clsx'

import { auth } from '@/lib/auth'
import { speakText, isSTTSupported } from '@/lib/speech'
import {
  getStatus, streamAnswer, verifyToken,
  listConversations, getConversation, createConversation,
  deleteConversation as apiDeleteConversation,
  renameConversation as apiRenameConversation,
} from '@/lib/api'
import type {
  Message, StatusResponse, Source, ConversationSummary, LLMMode,
} from '@/types'

import Sidebar        from '@/components/Sidebar'
import ChatMessage    from '@/components/ChatMessage'
import LLMModeToggle  from '@/components/LLMModeToggle'

// ── UUID generator ──────────────────────────────────────────────────────────
function newId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  return Math.random().toString(36).slice(2)
}

export default function ChatPage() {
  const router    = useRouter()
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef  = useRef<HTMLTextAreaElement>(null)

  // ── Core state ──────────────────────────────────────────────────────────
  const [token,           setToken]           = useState<string | null>(null)
  const [username,        setUsername]         = useState('')
  const [status,          setStatus]          = useState<StatusResponse | null>(null)
  const [messages,        setMessages]        = useState<Message[]>([])
  const [input,           setInput]           = useState('')
  const [streaming,       setStreaming]       = useState(false)

  // ── Conversation state ──────────────────────────────────────────────────
  const [conversationId,  setConversationId]  = useState<string | null>(null)
  const [conversations,   setConversations]   = useState<ConversationSummary[]>([])
  const [loadingConv,     setLoadingConv]     = useState(false)

  // ── LLM mode state ─────────────────────────────────────────────────────
  const [llmMode,         setLlmMode]         = useState<LLMMode>('auto')

  // ── Voice & Speech state ───────────────────────────────────────────────
  const [listening,       setListening]       = useState(false)
  const [autoRead,        setAutoRead]        = useState(false)
  const [sttSupported,    setSttSupported]    = useState(false)

  const recognitionRef = useRef<any>(null)
  const autoReadRef    = useRef(autoRead)
  autoReadRef.current  = autoRead

  useEffect(() => {
    setSttSupported(isSTTSupported())
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('docmind_auto_read') === 'true'
      setAutoRead(saved)
    }
  }, [])

  function toggleAutoRead() {
    const next = !autoRead
    setAutoRead(next)
    if (typeof window !== 'undefined') {
      localStorage.setItem('docmind_auto_read', String(next))
    }
  }

  function handleMicToggle() {
    if (listening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
      setListening(false)
      return
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) return

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'

    recognition.onstart = () => setListening(true)
    recognition.onend   = () => setListening(false)
    recognition.onerror = () => setListening(false)

    recognition.onresult = (event: any) => {
      let transcript = ''
      for (let i = 0; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript
      }
      if (transcript) setInput(transcript)
    }

    recognitionRef.current = recognition
    try {
      recognition.start()
    } catch {
      setListening(false)
    }
  }

  // ── Layout state ────────────────────────────────────────────────────────
  const [sidebarOpen,     setSidebarOpen]     = useState(true)

  // ── Auth gate ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const t = auth.getToken()
    if (!t) { router.replace('/auth'); return }

    verifyToken(t).then(valid => {
      if (!valid) { auth.clear(); router.replace('/auth'); return }
      setToken(t)
      setUsername(auth.getUsername() ?? '')
    })
  }, [router])

  // ── Load status + conversations on token ready ────────────────────────────
  const refreshStatus = useCallback(async () => {
    if (!token) return
    try {
      const s = await getStatus(token)
      setStatus(s)
    } catch {}
  }, [token])

  const refreshConversations = useCallback(async () => {
    if (!token) return
    try {
      const convs = await listConversations(token)
      setConversations(convs)
    } catch {}
  }, [token])

  useEffect(() => {
    refreshStatus()
    refreshConversations()
  }, [refreshStatus, refreshConversations])

  // ── Auto-scroll to bottom on new messages ─────────────────────────────────
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // ── Handle selecting an existing conversation ─────────────────────────────
  async function handleSelectConversation(id: string) {
    if (!token || id === conversationId) return
    setLoadingConv(true)
    try {
      const conv = await getConversation(id, token)
      // Map backend messages → frontend Message[]
      const mapped: Message[] = conv.messages.map(m => ({
        id:        m.id,
        role:      m.role,
        content:   m.content,
        sources:   m.sources ?? [],
        timestamp: new Date(m.created_at),
        provider:  (m.llm_provider_used as Message['provider']) ?? undefined,
      }))
      setMessages(mapped)
      setConversationId(id)
    } catch {
      // Conversation may have been deleted; refresh list
      refreshConversations()
    } finally {
      setLoadingConv(false)
    }
    // Close sidebar on mobile
    if (window.innerWidth < 768) setSidebarOpen(false)
  }

  // ── Handle creating a new chat ────────────────────────────────────────────
  async function handleNewChat() {
    if (!token) return
    try {
      const { id } = await createConversation(token)
      setConversationId(id)
      setMessages([])
      setInput('')
      refreshConversations()
      inputRef.current?.focus()
    } catch (err) {
      console.error('Failed to create conversation:', err)
    }
  }

  // ── Handle deleting a conversation ────────────────────────────────────────
  async function handleDeleteConversation(id: string) {
    if (!token) return
    try {
      await apiDeleteConversation(id, token)
      // If we deleted the active conversation, clear the chat
      if (id === conversationId) {
        setConversationId(null)
        setMessages([])
      }
      refreshConversations()
    } catch {}
  }

  // ── Handle renaming a conversation ────────────────────────────────────────
  async function handleRenameConversation(id: string, title: string) {
    if (!token) return
    try {
      await apiRenameConversation(id, title, token)
      refreshConversations()
    } catch {}
  }

  // ── Submit question ───────────────────────────────────────────────────────
  async function handleSubmit(e?: React.FormEvent) {
    e?.preventDefault()
    const question = input.trim()
    if (!question || !token || streaming) return
    if (!status?.ready) return

    setInput('')
    setStreaming(true)

    // Add user message
    const userMsg: Message = {
      id: newId(), role: 'user', content: question, sources: [], timestamp: new Date(),
    }

    // Add placeholder assistant message (streaming = true)
    const assistantId = newId()
    const assistantPlaceholder: Message = {
      id: assistantId, role: 'assistant', content: '', sources: [],
      isStreaming: true, timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMsg, assistantPlaceholder])

    // Stream the answer
    let   fullContent = ''
    let   finalSources: Source[] = []
    let   resolvedProvider: Message['provider'] = undefined

    await streamAnswer({
      question,
      conversationId: conversationId ?? undefined,
      llmMode,
      token,
      onToken(t) {
        fullContent += t
        // Update the placeholder message with each new token
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, content: fullContent } : m
        ))
      },
      onMetadata(sources, newConversationId, provider) {
        finalSources = sources
        resolvedProvider = provider as Message['provider']
        // If backend auto-created the conversation, adopt the ID
        if (newConversationId && newConversationId !== conversationId) {
          setConversationId(newConversationId)
        }
      },
      onError(msg) {
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? { ...m, content: `⚠️ ${msg}`, isStreaming: false, error: true }
            : m
        ))
      },
      onDone() {
        // Finalise the assistant message: mark streaming done, attach sources + provider
        setMessages(prev => prev.map(m =>
          m.id === assistantId
            ? {
                ...m,
                content: fullContent || m.content,
                sources: finalSources,
                isStreaming: false,
                provider: resolvedProvider,
              }
            : m
        ))
        setStreaming(false)
        refreshConversations()
        inputRef.current?.focus()

        // Auto-read hands-free feature
        if (autoReadRef.current && fullContent) {
          speakText(fullContent)
        }
      },
    })
  }

  // ── Handle Enter key (Shift+Enter = newline) ──────────────────────────────
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  // ── Sign out ──────────────────────────────────────────────────────────────
  function handleSignOut() {
    auth.clear()
    router.replace('/auth')
  }

  // ── Loading state ─────────────────────────────────────────────────────────
  if (!token) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const canAsk     = status?.ready && !streaming
  const hasMessages = messages.length > 0

  return (
    <div className="h-screen flex overflow-hidden bg-slate-50">

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <Sidebar
        status={status}
        token={token}
        username={username}
        conversations={conversations}
        activeConversationId={conversationId}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(v => !v)}
        onUpload={refreshStatus}
        onNewChat={handleNewChat}
        onSignOut={handleSignOut}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        onRenameConversation={handleRenameConversation}
      />

      {/* ── Main chat area ───────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0">

        {/* Top bar */}
        <header className="h-14 shrink-0 border-b border-slate-200 bg-white
                           flex items-center px-4 md:px-6 gap-3">
          {/* Hamburger toggle */}
          <button
            onClick={() => setSidebarOpen(v => !v)}
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-700
                       hover:bg-slate-100 transition-colors"
            title="Toggle sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>

          <Brain className="w-5 h-5 text-indigo-500" />
          <span className="font-semibold text-slate-800">Document Q&A</span>

          {streaming && (
            <span className="ml-auto flex items-center gap-2 text-xs text-indigo-600 font-medium">
              <span className="typing-dots flex gap-0.5">
                <span /><span /><span />
              </span>
              Generating…
            </span>
          )}
        </header>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto px-4 md:px-6 py-6 space-y-6">
          {/* Empty state — no conversation selected or no messages */}
          {!hasMessages && !loadingConv && (
            <div className="flex-1 flex flex-col items-center justify-center text-center
                            py-20 animate-fade-in">
              <div className="w-16 h-16 bg-indigo-100 rounded-2xl flex items-center
                              justify-center mb-4">
                <Brain className="w-8 h-8 text-indigo-500" />
              </div>
              <h2 className="text-lg font-semibold text-slate-800 mb-2">
                {status?.ready
                  ? 'Ask anything about your documents'
                  : 'Welcome to DocMind'}
              </h2>
              <p className="text-sm text-slate-500 max-w-md">
                {status?.ready
                  ? 'Start a conversation below. Your answers are grounded entirely in your uploaded documents — no hallucinations.'
                  : 'Upload a PDF document in the sidebar to get started. I\'ll answer questions using only what\'s in your documents.'}
              </p>
              {!conversationId && status?.ready && (
                <button
                  onClick={handleNewChat}
                  className="mt-6 btn-primary"
                >
                  <MessageSquarePlus className="w-4 h-4" />
                  Start a conversation
                </button>
              )}
            </div>
          )}

          {/* Loading conversation */}
          {loadingConv && (
            <div className="flex items-center justify-center py-20">
              <div className="w-5 h-5 border-2 border-indigo-600 border-t-transparent
                              rounded-full animate-spin" />
            </div>
          )}

          {/* Rendered messages */}
          {hasMessages && messages.map(msg => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input bar */}
        <div className="shrink-0 border-t border-slate-200 bg-white px-4 md:px-6 py-4">
          {!status?.ready && (
            <p className="text-xs text-center text-amber-600 bg-amber-50 rounded-lg
                           px-3 py-2 mb-3">
              Upload a PDF document in the sidebar to start asking questions.
            </p>
          )}
          <form onSubmit={handleSubmit} className="flex items-end gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!canAsk}
                placeholder={
                  !status?.ready
                    ? 'Upload a document first…'
                    : !conversationId
                    ? 'Start a new chat to ask questions…'
                    : 'Ask a question about your documents… (Enter to send)'
                }
                rows={1}
                className={clsx(
                  'w-full resize-none input py-3 pr-4 max-h-36',
                  'disabled:bg-slate-50 disabled:cursor-not-allowed',
                )}
                style={{ minHeight: '48px' }}
              />
            </div>

            {/* Voice microphone button */}
            <button
              type="button"
              onClick={handleMicToggle}
              disabled={!canAsk || !sttSupported}
              className={clsx(
                "h-12 w-12 rounded-xl flex items-center justify-center shrink-0 border transition-all",
                listening
                  ? "bg-red-50 border-red-300 text-red-600 animate-pulse shadow-sm"
                  : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100 hover:text-indigo-600",
                (!canAsk || !sttSupported) && "opacity-40 cursor-not-allowed"
              )}
              title={
                !sttSupported
                  ? "Voice input not supported in this browser (Chrome recommended)"
                  : listening
                  ? "Listening... Click to stop"
                  : "Speak question (Voice Input)"
              }
            >
              {listening ? <MicOff className="w-4 h-4 text-red-600" /> : <Mic className="w-4 h-4" />}
            </button>

            <button
              type="submit"
              disabled={!canAsk || !input.trim() || !conversationId}
              className="btn-primary h-12 w-12 p-0 rounded-xl shrink-0"
              title="Send (Enter)"
            >
              {streaming
                ? <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                : <Send className="w-4 h-4" />}
            </button>
          </form>

          {/* LLM mode toggle + auto read toggle + footer text */}
          <div className="flex items-center justify-between mt-3 flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <LLMModeToggle
                mode={llmMode}
                onChange={setLlmMode}
                disabled={streaming}
              />
              <button
                type="button"
                onClick={toggleAutoRead}
                className={clsx(
                  "flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors",
                  autoRead
                    ? "bg-indigo-50 border-indigo-200 text-indigo-700"
                    : "bg-slate-50 border-slate-200 text-slate-500 hover:text-slate-700"
                )}
                title="Automatically speak assistant responses aloud"
              >
                {autoRead ? <Volume2 className="w-3 h-3 text-indigo-600" /> : <VolumeX className="w-3 h-3 text-slate-400" />}
                <span>Auto-read</span>
              </button>
            </div>

            <p className="text-xs text-slate-400 hidden sm:block">
              Answers grounded in your documents only
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}
