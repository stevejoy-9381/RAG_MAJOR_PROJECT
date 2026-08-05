'use client'

/* eslint-disable react/prop-types */

import { useState, useEffect } from 'react'
import { Brain, User, Cpu, Zap, Volume2, Square } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '@/types'
import SourcePanel from './SourcePanel'
import { clsx } from 'clsx'
import { speakText, stopSpeech, isTTSSupported } from '@/lib/speech'

interface Props {
  message: Message
  onRetry?: () => void
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** Provider badge shown on assistant messages */
function ProviderBadge({ provider }: { provider: string }) {
  const isOllama = provider === 'ollama'
  return (
    <span className={clsx(
      'inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full',
      isOllama
        ? 'bg-emerald-50 text-emerald-600'
        : 'bg-violet-50 text-violet-600',
    )}>
      {isOllama
        ? <Cpu className="w-2.5 h-2.5" />
        : <Zap className="w-2.5 h-2.5" />}
      {provider}
    </span>
  )
}

export default function ChatMessage({ message, onRetry }: Props) {
  const isUser = message.role === 'user'
  const [speaking, setSpeaking] = useState(false)
  const ttsAvailable = isTTSSupported()

  useEffect(() => {
    return () => {
      if (speaking) stopSpeech()
    }
  }, [speaking])

  function handleToggleSpeech() {
    if (speaking) {
      stopSpeech()
      setSpeaking(false)
    } else {
      setSpeaking(true)
      speakText(
        message.content,
        () => setSpeaking(false),
        () => setSpeaking(false)
      )
    }
  }

  return (
    <div className={clsx(
      'flex gap-3 message-enter',
      isUser ? 'flex-row-reverse' : 'flex-row'
    )}>
      {/* Avatar */}
      <div className={clsx(
        'w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-0.5',
        isUser
          ? 'bg-indigo-600 text-white'
          : 'bg-white border-2 border-slate-200 text-indigo-600'
      )}>
        {isUser
          ? <User  className="w-4 h-4" />
          : <Brain className="w-4 h-4" />}
      </div>

      {/* Bubble + sources */}
      <div className={clsx(
        'flex flex-col max-w-[75%]',
        isUser ? 'items-end' : 'items-start'
      )}>
        <div className={clsx(
          'px-4 py-3 rounded-2xl text-sm leading-relaxed',
          isUser
            ? 'bg-indigo-600 text-white rounded-tr-sm'
            : clsx(
                'bg-white border border-slate-200 text-slate-800 rounded-tl-sm shadow-sm',
                message.error && 'border-red-200 bg-red-50 text-red-700',
              )
        )}>
          {/* Message content */}
          {isUser ? (
            <span>{message.content || '…'}</span>
          ) : (
            <div className={clsx(
              'prose-assistant',
              message.isStreaming && !message.error && 'streaming-cursor',
            )}>
              {message.content ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    pre: ({ children }) => (
                      <pre className="bg-slate-800 text-slate-100 rounded-lg p-3 overflow-x-auto text-xs my-2 font-mono">
                        {children}
                      </pre>
                    ),
                    code: ({ children, className }) => {
                      const isBlock = className?.includes('language-')
                      if (isBlock) {
                        return <code className="text-xs">{children}</code>
                      }
                      return (
                        <code className="bg-slate-100 text-indigo-700 px-1 py-0.5 rounded text-xs font-mono">
                          {children}
                        </code>
                      )
                    },
                    table: ({ children }) => (
                      <div className="overflow-x-auto my-2">
                        <table className="text-xs border-collapse w-full">
                          {children}
                        </table>
                      </div>
                    ),
                    th: ({ children }) => (
                      <th className="border border-slate-200 bg-slate-50 px-2 py-1 text-left text-xs font-semibold">
                        {children}
                      </th>
                    ),
                    td: ({ children }) => (
                      <td className="border border-slate-200 px-2 py-1 text-xs">
                        {children}
                      </td>
                    ),
                    a: ({ children, href }) => (
                      <a href={href} target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:text-indigo-800 underline">
                        {children}
                      </a>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              ) : (
                message.isStreaming ? '' : '…'
              )}
            </div>
          )}
        </div>

        {/* Source panel (assistant only, after streaming completes) */}
        {!isUser && !message.isStreaming && message.sources && message.sources.length > 0 && (
          <div className="w-full max-w-lg mt-1">
            <SourcePanel sources={message.sources} />
          </div>
        )}

        {/* Timestamp + provider badge + read aloud + retry button */}
        <div className="flex items-center gap-2 mt-1 px-1">
          <span className="text-xs text-slate-400">
            {formatTime(message.timestamp)}
          </span>
          {!isUser && message.provider && !message.isStreaming && (
            <ProviderBadge provider={message.provider} />
          )}
          {!isUser && message.error && onRetry && (
            <button
              onClick={onRetry}
              className="text-xs font-medium text-red-600 hover:text-red-700 hover:underline flex items-center gap-1"
            >
              🔄 Retry
            </button>
          )}
          {!isUser && !message.isStreaming && message.content && ttsAvailable && !message.error && (
            <button
              onClick={handleToggleSpeech}
              className={clsx(
                "p-1 rounded-md transition-colors text-xs flex items-center gap-1",
                speaking
                  ? "bg-indigo-100 text-indigo-700 font-medium"
                  : "text-slate-400 hover:text-slate-600 hover:bg-slate-100"
              )}
              title={speaking ? "Stop reading aloud" : "Read message aloud"}
            >
              {speaking ? (
                <>
                  <Square className="w-3 h-3 fill-indigo-600" />
                  <span className="text-[10px]">Stop</span>
                </>
              ) : (
                <Volume2 className="w-3.5 h-3.5" />
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
