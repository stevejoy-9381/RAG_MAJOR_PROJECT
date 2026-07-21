'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import {
  ArrowLeft, BarChart2, MessageSquare, Clock, FileText,
  Zap, Database, RefreshCw, AlertCircle, HelpCircle, Layers
} from 'lucide-react'
import { clsx } from 'clsx'

import { auth } from '@/lib/auth'
import { getAnalyticsSummary } from '@/lib/api'
import type { AnalyticsSummary } from '@/types'

export default function AnalyticsPage() {
  const router = useRouter()

  const [token, setToken] = useState<string | null>(null)
  const [data, setData] = useState<AnalyticsSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const t = auth.getToken()
    if (!t) {
      router.push('/auth')
      return
    }
    setToken(t)
    fetchAnalytics(t)
  }, [router])

  async function fetchAnalytics(t: string) {
    setLoading(true)
    setError(null)
    try {
      const summary = await getAnalyticsSummary(t)
      setData(summary)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analytics.')
    } finally {
      setLoading(false)
    }
  }

  const maxDailyQuestions = data?.usage_over_time
    ? Math.max(...data.usage_over_time.map(d => d.questions), 1)
    : 1

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Navbar */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <Link
            href="/chat"
            className="p-2 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors flex items-center gap-1.5 text-sm font-medium"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Chat
          </Link>
          <div className="h-5 w-px bg-slate-200" />
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-emerald-600 rounded-lg flex items-center justify-center">
              <BarChart2 className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-slate-900 text-base leading-none">Usage & Insights Analytics</h1>
              <p className="text-xs text-slate-500 mt-0.5">Track question frequency, document citations, latency, and LLM distribution</p>
            </div>
          </div>
        </div>

        <button
          onClick={() => token && fetchAnalytics(token)}
          disabled={loading}
          className="p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors text-xs flex items-center gap-1.5 font-medium"
          title="Refresh analytics data"
        >
          <RefreshCw className={clsx("w-3.5 h-3.5", loading && "animate-spin")} />
          Refresh
        </button>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm flex items-start gap-2.5">
            <AlertCircle className="w-5 h-5 shrink-0 text-red-500 mt-0.5" />
            <div className="flex-1">{error}</div>
          </div>
        )}

        {loading ? (
          <div className="py-20 text-center text-slate-400 space-y-3">
            <RefreshCw className="w-8 h-8 mx-auto animate-spin text-emerald-500" />
            <p className="text-sm">Calculating usage analytics and document citation metrics...</p>
          </div>
        ) : data ? (
          <>
            {/* Top KPI Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <div className="flex items-center justify-between text-slate-500 mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider">Conversations</span>
                  <MessageSquare className="w-4 h-4 text-indigo-500" />
                </div>
                <div className="text-3xl font-bold text-slate-900">{data.total_conversations}</div>
                <p className="text-xs text-slate-400 mt-1">Total active sessions created</p>
              </div>

              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <div className="flex items-center justify-between text-slate-500 mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider">Questions Answered</span>
                  <Zap className="w-4 h-4 text-emerald-500" />
                </div>
                <div className="text-3xl font-bold text-slate-900">{data.total_answers}</div>
                <p className="text-xs text-slate-400 mt-1">Total assistant responses generated</p>
              </div>

              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <div className="flex items-center justify-between text-slate-500 mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider">Avg Response Time</span>
                  <Clock className="w-4 h-4 text-amber-500" />
                </div>
                <div className="text-3xl font-bold text-slate-900">
                  {data.avg_latency_seconds > 0 ? `${data.avg_latency_seconds.toFixed(2)}s` : 'N/A'}
                </div>
                <p className="text-xs text-slate-400 mt-1">Across all streaming/chat requests</p>
              </div>

              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <div className="flex items-center justify-between text-slate-500 mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider">Most Cited Doc</span>
                  <FileText className="w-4 h-4 text-blue-500" />
                </div>
                <div className="text-sm font-bold text-slate-900 truncate" title={data.most_cited_documents[0]?.document || 'None'}>
                  {data.most_cited_documents[0]?.document || 'No citations yet'}
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  {data.most_cited_documents[0]
                    ? `${data.most_cited_documents[0].citations} total citations`
                    : 'Upload & query docs to see insights'}
                </p>
              </div>
            </div>

            {/* Middle Grid: Usage over time + Provider split */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Usage over time chart (2 Cols) */}
              <div className="lg:col-span-2 bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div>
                    <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                      <BarChart2 className="w-4 h-4 text-emerald-600" />
                      Usage Activity Over Time (14 Days)
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">Daily questions asked by user</p>
                  </div>
                </div>

                <div className="h-44 flex items-end justify-between gap-1.5 pt-4 px-2">
                  {data.usage_over_time.map((dp) => {
                    const heightPercent = maxDailyQuestions > 0 ? (dp.questions / maxDailyQuestions) * 100 : 0
                    const formattedDate = dp.date.slice(5) // MM-DD
                    return (
                      <div key={dp.date} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end group">
                        <div className="text-[10px] font-semibold text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity">
                          {dp.questions}
                        </div>
                        <div
                          style={{ height: `${Math.max(heightPercent, 4)}%` }}
                          className={clsx(
                            "w-full rounded-t-lg transition-all",
                            dp.questions > 0 ? "bg-emerald-500 group-hover:bg-emerald-600 shadow-sm" : "bg-slate-100"
                          )}
                        />
                        <span className="text-[10px] text-slate-400 truncate w-full text-center">{formattedDate}</span>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Provider Split Donut / Progress Breakdown (1 Col) */}
              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4 flex flex-col justify-between">
                <div>
                  <div className="border-b border-slate-100 pb-3">
                    <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-amber-500" />
                      LLM Provider Split
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">Online (Groq) vs Offline (Ollama)</p>
                  </div>

                  <div className="py-6 space-y-5">
                    {/* Progress Bar */}
                    <div className="h-4 w-full bg-slate-100 rounded-full overflow-hidden flex">
                      {data.provider_split.total > 0 ? (
                        <>
                          <div
                            style={{ width: `${(data.provider_split.offline / data.provider_split.total) * 100}%` }}
                            className="bg-indigo-500 h-full transition-all"
                            title={`Offline (Ollama): ${data.provider_split.offline}`}
                          />
                          <div
                            style={{ width: `${(data.provider_split.online / data.provider_split.total) * 100}%` }}
                            className="bg-emerald-500 h-full transition-all"
                            title={`Online (Groq): ${data.provider_split.online}`}
                          />
                          <div
                            style={{ width: `${(data.provider_split.other / data.provider_split.total) * 100}%` }}
                            className="bg-slate-300 h-full transition-all"
                            title={`Other: ${data.provider_split.other}`}
                          />
                        </>
                      ) : (
                        <div className="w-full bg-slate-200 h-full" />
                      )}
                    </div>

                    <div className="space-y-2 text-xs">
                      <div className="flex items-center justify-between p-2.5 rounded-xl bg-indigo-50/60 border border-indigo-100">
                        <span className="flex items-center gap-2 font-medium text-indigo-950">
                          <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 inline-block" />
                          Offline (Ollama)
                        </span>
                        <span className="font-bold text-indigo-900">
                          {data.provider_split.offline} ({data.provider_split.total > 0 ? Math.round((data.provider_split.offline / data.provider_split.total) * 100) : 0}%)
                        </span>
                      </div>

                      <div className="flex items-center justify-between p-2.5 rounded-xl bg-emerald-50/60 border border-emerald-100">
                        <span className="flex items-center gap-2 font-medium text-emerald-950">
                          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
                          Online (Groq)
                        </span>
                        <span className="font-bold text-emerald-900">
                          {data.provider_split.online} ({data.provider_split.total > 0 ? Math.round((data.provider_split.online / data.provider_split.total) * 100) : 0}%)
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 text-center border-t border-slate-100 pt-3">
                  Total LLM calls logged: <span className="font-semibold text-slate-600">{data.provider_split.total}</span>
                </div>
              </div>
            </div>

            {/* Bottom Grid: Top Questions + Most Cited Docs */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Top Questions */}
              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
                <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <HelpCircle className="w-4 h-4 text-indigo-600" />
                    Top Asked Questions
                  </h2>
                  <span className="text-xs text-slate-400">Frequency</span>
                </div>

                {data.top_questions.length === 0 ? (
                  <div className="py-8 text-center text-slate-400 text-xs italic">
                    No questions recorded yet. Start asking questions in Chat!
                  </div>
                ) : (
                  <div className="space-y-2">
                    {data.top_questions.map((q, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100 text-xs hover:bg-slate-100/70 transition-colors"
                      >
                        <div className="flex items-start gap-2.5 min-w-0">
                          <span className="font-bold text-slate-400 shrink-0 w-4 text-right">#{idx + 1}</span>
                          <span className="text-slate-800 font-medium truncate" title={q.question}>
                            {q.question}
                          </span>
                        </div>
                        <span className="font-bold bg-indigo-50 text-indigo-700 px-2.5 py-0.5 rounded-full text-xs shrink-0">
                          {q.count} {q.count === 1 ? 'time' : 'times'}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Most Cited Documents */}
              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
                <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                    <Database className="w-4 h-4 text-blue-600" />
                    Most Cited Documents
                  </h2>
                  <span className="text-xs text-slate-400">Citations</span>
                </div>

                {data.most_cited_documents.length === 0 ? (
                  <div className="py-8 text-center text-slate-400 text-xs italic">
                    No document citations recorded yet. Upload files and query them!
                  </div>
                ) : (
                  <div className="space-y-2">
                    {data.most_cited_documents.map((d, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between gap-3 p-3 rounded-xl bg-slate-50 border border-slate-100 text-xs hover:bg-slate-100/70 transition-colors"
                      >
                        <div className="flex items-center gap-2.5 min-w-0">
                          <FileText className="w-4 h-4 text-blue-500 shrink-0" />
                          <div className="min-w-0">
                            <p className="font-medium text-slate-800 truncate" title={d.document}>
                              {d.document}
                            </p>
                            <p className="text-[11px] text-slate-400 mt-0.5">
                              {d.unique_pages_cited} unique page{d.unique_pages_cited !== 1 ? 's' : ''} cited
                            </p>
                          </div>
                        </div>
                        <span className="font-bold bg-blue-50 text-blue-700 px-2.5 py-0.5 rounded-full text-xs shrink-0">
                          {d.citations} {d.citations === 1 ? 'citation' : 'citations'}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        ) : null}
      </main>
    </div>
  )
}
