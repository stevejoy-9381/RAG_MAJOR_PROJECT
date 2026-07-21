'use client'

import { clsx } from 'clsx'
import { Zap, Monitor, Cpu } from 'lucide-react'
import type { LLMMode } from '@/types'

interface Props {
  mode:     LLMMode
  onChange: (mode: LLMMode) => void
  disabled?: boolean
}

const modes: { value: LLMMode; label: string; icon: typeof Zap; tip: string }[] = [
  { value: 'auto',    label: 'Auto',    icon: Zap,     tip: 'Automatically pick the best available provider' },
  { value: 'offline', label: 'Offline', icon: Cpu,     tip: 'Force local Ollama model' },
  { value: 'online',  label: 'Online',  icon: Monitor, tip: 'Force cloud Groq API' },
]

export default function LLMModeToggle({ mode, onChange, disabled }: Props) {
  return (
    <div className="inline-flex items-center bg-slate-100 rounded-lg p-0.5 gap-0.5">
      {modes.map(m => {
        const Icon     = m.icon
        const isActive = mode === m.value

        return (
          <button
            key={m.value}
            onClick={() => onChange(m.value)}
            disabled={disabled}
            title={m.tip}
            className={clsx(
              'flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium',
              'transition-all duration-150',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              isActive
                ? 'bg-white text-indigo-700 shadow-sm ring-1 ring-slate-200'
                : 'text-slate-500 hover:text-slate-700 hover:bg-slate-50',
            )}
          >
            <Icon className="w-3 h-3" />
            {m.label}
          </button>
        )
      })}
    </div>
  )
}
