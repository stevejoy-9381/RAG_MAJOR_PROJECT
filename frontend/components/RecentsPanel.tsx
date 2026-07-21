'use client'

import { useState, useRef, useEffect } from 'react'
import {
  MessageSquare, Trash2, Pencil, Check, X, MessageCircle,
} from 'lucide-react'
import { clsx } from 'clsx'
import type { ConversationSummary } from '@/types'

interface Props {
  conversations:        ConversationSummary[]
  activeConversationId: string | null
  onSelect:             (id: string) => void
  onDelete:             (id: string) => void
  onRename:             (id: string, title: string) => void
}

// ── Relative time helper ─────────────────────────────────────────────────────

function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString)
  const now  = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSec = Math.floor(diffMs / 1000)

  if (diffSec < 60)    return 'just now'
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60)    return `${diffMin}m ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24)     return `${diffHr}h ago`
  const diffDay = Math.floor(diffHr / 24)
  if (diffDay === 1)   return 'yesterday'
  if (diffDay < 7)     return `${diffDay}d ago`
  const diffWeek = Math.floor(diffDay / 7)
  if (diffWeek < 4)    return `${diffWeek}w ago`
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

// ── Component ────────────────────────────────────────────────────────────────

export default function RecentsPanel({
  conversations, activeConversationId, onSelect, onDelete, onRename,
}: Props) {
  const [editingId, setEditingId]       = useState<string | null>(null)
  const [editTitle, setEditTitle]       = useState('')
  const editInputRef = useRef<HTMLInputElement>(null)

  // Focus the rename input when it appears
  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus()
      editInputRef.current.select()
    }
  }, [editingId])

  function startRename(conv: ConversationSummary) {
    setEditingId(conv.id)
    setEditTitle(conv.title)
  }

  function commitRename() {
    if (editingId && editTitle.trim()) {
      onRename(editingId, editTitle.trim())
    }
    setEditingId(null)
    setEditTitle('')
  }

  function cancelRename() {
    setEditingId(null)
    setEditTitle('')
  }

  function handleDelete(id: string, title: string) {
    if (confirm(`Delete "${title}"?`)) {
      onDelete(id)
    }
  }

  return (
    <div className="px-5 pt-4 pb-2">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Recents
        </h3>
        <span className="text-xs text-slate-400">{conversations.length}</span>
      </div>

      {conversations.length === 0 ? (
        <div className="text-center py-6 text-slate-400">
          <MessageCircle className="w-8 h-8 mx-auto mb-2 opacity-40" />
          <p className="text-xs">No conversations yet.<br />Start one below!</p>
        </div>
      ) : (
        <div className="space-y-0.5">
          {conversations.map(conv => {
            const isActive  = conv.id === activeConversationId
            const isEditing = conv.id === editingId

            return (
              <div
                key={conv.id}
                className={clsx(
                  'group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer',
                  'transition-all duration-150',
                  isActive
                    ? 'bg-indigo-50 text-indigo-700 border border-indigo-100'
                    : 'hover:bg-slate-50 text-slate-700 border border-transparent',
                )}
                onClick={() => { if (!isEditing) onSelect(conv.id) }}
              >
                <MessageSquare className={clsx(
                  'w-3.5 h-3.5 shrink-0 mt-0.5',
                  isActive ? 'text-indigo-500' : 'text-slate-400',
                )} />

                <div className="flex-1 min-w-0">
                  {isEditing ? (
                    <div className="flex items-center gap-1">
                      <input
                        ref={editInputRef}
                        value={editTitle}
                        onChange={e => setEditTitle(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter') commitRename()
                          if (e.key === 'Escape') cancelRename()
                        }}
                        onBlur={commitRename}
                        onClick={e => e.stopPropagation()}
                        className="w-full text-xs font-medium bg-white border border-indigo-300
                                   rounded px-1.5 py-0.5 focus:outline-none focus:ring-1
                                   focus:ring-indigo-400 text-slate-800"
                      />
                      <button
                        onClick={e => { e.stopPropagation(); commitRename() }}
                        className="p-0.5 text-green-600 hover:text-green-700"
                      >
                        <Check className="w-3 h-3" />
                      </button>
                      <button
                        onClick={e => { e.stopPropagation(); cancelRename() }}
                        className="p-0.5 text-slate-400 hover:text-slate-600"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <p className="text-xs font-medium truncate">{conv.title}</p>
                      <p className={clsx(
                        'text-[10px] mt-0.5 truncate',
                        isActive ? 'text-indigo-400' : 'text-slate-400',
                      )}>
                        {conv.last_message_preview || 'Empty conversation'}
                        {' · '}
                        {formatRelativeTime(conv.updated_at)}
                      </p>
                    </>
                  )}
                </div>

                {/* Action buttons — only show on hover, never during edit */}
                {!isEditing && (
                  <div className="flex items-center gap-0.5 shrink-0 opacity-0
                                  group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={e => { e.stopPropagation(); startRename(conv) }}
                      className={clsx(
                        'p-1 rounded transition-colors',
                        isActive
                          ? 'text-indigo-400 hover:text-indigo-600 hover:bg-indigo-100'
                          : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100',
                      )}
                      title="Rename"
                    >
                      <Pencil className="w-3 h-3" />
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); handleDelete(conv.id, conv.title) }}
                      className="p-1 rounded text-slate-400 hover:text-red-500
                                 hover:bg-red-50 transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
