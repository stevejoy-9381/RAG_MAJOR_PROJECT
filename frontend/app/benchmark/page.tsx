'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import {
  ArrowLeft, Gauge, Play, RefreshCw, CheckCircle2,
  Clock, Cpu, FileText, AlertCircle, ChevronDown, ChevronUp, Layers
} from 'lucide-react'
import { clsx } from 'clsx'

import { auth } from '@/lib/auth'
import { listBenchmarkModels, runBenchmark, getBenchmarkResults } from '@/lib/api'
import type { BenchmarkGroup, BenchmarkItem } from '@/types'

const PRESET_QUESTIONS = [
  "What are the key findings or main conclusions in the uploaded documents?",
  "Summarize the methodology or process described in the documents.",
  "List any important metrics, numbers, or specifications mentioned."
]

export default function BenchmarkPage() {
  const router = useRouter()

  const [token, setToken] = useState<string | null>(null)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [questionsInput, setQuestionsInput] = useState(PRESET_QUESTIONS.join('\n'))

  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [currentRun, setCurrentRun] = useState<BenchmarkGroup | null>(null)
  const [pastRuns, setPastRuns] = useState<BenchmarkGroup[]>([])
  const [loadingModels, setLoadingModels] = useState(true)

  const [sortField, setSortField] = useState<'latency' | 'model'>('latency')
  const [sortAsc, setSortAsc] = useState(true)

  const [expandedAnswerId, setExpandedAnswerId] = useState<string | null>(null)

  useEffect(() => {
    const t = auth.getToken()
    if (!t) {
      router.push('/auth')
      return
    }
    setToken(t)
    fetchInitialData(t)
  }, [router])

  async function fetchInitialData(t: string) {
    setLoadingModels(true)
    setError(null)
    try {
      const [modelsRes, resultsRes] = await Promise.all([
        listBenchmarkModels(t),
        getBenchmarkResults(t),
      ])

      const models = modelsRes.models || []
      setAvailableModels(models)
      // Select first two models by default if available
      if (models.length > 0) {
        setSelectedModels(models.slice(0, Math.min(2, models.length)))
      }

      setPastRuns(resultsRes || [])
      if (resultsRes && resultsRes.length > 0) {
        setCurrentRun(resultsRes[0])
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load benchmark data.')
    } finally {
      setLoadingModels(false)
    }
  }

  function handleModelToggle(model: string) {
    if (selectedModels.includes(model)) {
      setSelectedModels(selectedModels.filter(m => m !== model))
    } else {
      setSelectedModels([...selectedModels, model])
    }
  }

  async function handleRunBenchmark() {
    if (!token) return
    if (selectedModels.length === 0) {
      setError('Please select at least one Ollama model to benchmark.')
      return
    }

    const questions = questionsInput
      .split('\n')
      .map(q => q.strip ? q.strip() : q.trim())
      .filter(q => q.length > 0)

    if (questions.length === 0) {
      setError('Please enter at least one question.')
      return
    }

    setRunning(true)
    setError(null)

    try {
      const result = await runBenchmark(selectedModels, questions, token)
      setCurrentRun(result)
      setPastRuns(prev => [result, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed.')
    } finally {
      setRunning(false)
    }
  }

  // Active run data to display
  const activeRun = currentRun

  const sortedResults = activeRun ? [...activeRun.results].sort((a, b) => {
    if (sortField === 'latency') {
      return sortAsc ? a.latency_seconds - b.latency_seconds : b.latency_seconds - a.latency_seconds
    } else {
      return sortAsc ? a.model.localeCompare(b.model) : b.model.localeCompare(a.model)
    }
  }) : []

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
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <Gauge className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-slate-900 text-base leading-none">Model Benchmark</h1>
              <p className="text-xs text-slate-500 mt-0.5">Compare local LLM speed & response quality on your documents</p>
            </div>
          </div>
        </div>

        <button
          onClick={() => token && fetchInitialData(token)}
          disabled={loadingModels || running}
          className="p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 transition-colors text-xs flex items-center gap-1.5 font-medium"
          title="Refresh available models and history"
        >
          <RefreshCw className={clsx("w-3.5 h-3.5", (loadingModels || running) && "animate-spin")} />
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

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls Panel (Left Column) */}
          <div className="bg-white border border-slate-200 rounded-2xl p-5 space-y-5 shadow-sm h-fit">
            <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-indigo-600" />
              Configure Test Run
            </h2>

            {/* Model Selection */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-2 uppercase tracking-wider">
                Select Local Ollama Models ({selectedModels.length} selected)
              </label>

              {loadingModels ? (
                <div className="text-xs text-slate-400 py-3 text-center animate-pulse">
                  Scanning for local Ollama models...
                </div>
              ) : availableModels.length === 0 ? (
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-800 space-y-1">
                  <p className="font-semibold">No Ollama models detected.</p>
                  <p>Ensure Ollama is running (`ollama serve`) and has models pulled (e.g. `ollama pull llama3.1:8b`).</p>
                </div>
              ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                  {availableModels.map(m => (
                    <label
                      key={m}
                      className={clsx(
                        "flex items-center gap-2.5 p-2.5 rounded-xl border text-xs cursor-pointer transition-all",
                        selectedModels.includes(m)
                          ? "bg-indigo-50/70 border-indigo-300 text-indigo-950 font-medium"
                          : "bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100"
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={selectedModels.includes(m)}
                        onChange={() => handleModelToggle(m)}
                        className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-4 h-4"
                      />
                      <span className="truncate">{m}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Questions Input */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Test Questions (1 per line)
                </label>
                <button
                  type="button"
                  onClick={() => setQuestionsInput(PRESET_QUESTIONS.join('\n'))}
                  className="text-xs text-indigo-600 hover:underline font-medium"
                >
                  Reset presets
                </button>
              </div>
              <textarea
                value={questionsInput}
                onChange={e => setQuestionsInput(e.target.value)}
                rows={5}
                placeholder="Enter test questions, one per line..."
                className="w-full text-xs p-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono resize-y"
              />
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunBenchmark}
              disabled={running || selectedModels.length === 0 || loadingModels}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-indigo-600 text-white rounded-xl font-medium text-sm hover:bg-indigo-700 active:bg-indigo-800 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-indigo-100 transition-colors"
            >
              {running ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Benchmark...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  Run Benchmark
                </>
              )}
            </button>

            {running && (
              <p className="text-xs text-center text-slate-500 italic">
                Executing retrieval once per question, then generating responses across models...
              </p>
            )}
          </div>

          {/* Results Area (Right 2 Columns) */}
          <div className="lg:col-span-2 space-y-6">
            {activeRun ? (
              <>
                {/* Summary Cards */}
                <div>
                  <h2 className="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-2">
                    <Clock className="w-4 h-4 text-indigo-600" />
                    Model Performance Summary
                  </h2>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(activeRun.summary).map(([modelName, stats]) => (
                      <div
                        key={modelName}
                        className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm relative overflow-hidden"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-semibold text-slate-800 truncate max-w-[140px]" title={modelName}>
                            {modelName}
                          </span>
                          <span className="text-[10px] bg-slate-100 text-slate-600 font-medium px-2 py-0.5 rounded-full">
                            {stats.total_questions} Qs
                          </span>
                        </div>
                        <div className="text-2xl font-bold text-indigo-600 mb-1">
                          {stats.avg_latency.toFixed(2)}s
                          <span className="text-xs text-slate-400 font-normal ml-1">avg</span>
                        </div>
                        {stats.min_latency !== undefined && stats.max_latency !== undefined && (
                          <p className="text-[11px] text-slate-400">
                            Min: {stats.min_latency.toFixed(1)}s · Max: {stats.max_latency.toFixed(1)}s
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Results Table */}
                <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
                  <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                      <FileText className="w-4 h-4 text-indigo-600" />
                      Detailed Benchmark Output ({sortedResults.length} responses)
                    </h2>
                    <span className="text-xs text-slate-400">
                      Run ID: {activeRun.run_group_id.slice(0, 8)}
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border-collapse">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-100 text-slate-500 font-medium">
                          <th
                            onClick={() => {
                              if (sortField === 'model') setSortAsc(!sortAsc)
                              else { setSortField('model'); setSortAsc(true) }
                            }}
                            className="py-3 px-4 cursor-pointer hover:text-slate-900 select-none"
                          >
                            Model {sortField === 'model' && (sortAsc ? '▲' : '▼')}
                          </th>
                          <th className="py-3 px-4">Question</th>
                          <th
                            onClick={() => {
                              if (sortField === 'latency') setSortAsc(!sortAsc)
                              else { setSortField('latency'); setSortAsc(true) }
                            }}
                            className="py-3 px-4 cursor-pointer hover:text-slate-900 select-none text-right"
                          >
                            Latency {sortField === 'latency' && (sortAsc ? '▲' : '▼')}
                          </th>
                          <th className="py-3 px-4 text-right">Tokens</th>
                          <th className="py-3 px-4 text-center">Answer</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {sortedResults.map((item: BenchmarkItem) => {
                          const isExpanded = expandedAnswerId === item.id
                          return (
                            <tr key={item.id} className="hover:bg-slate-50/70 transition-colors">
                              <td className="py-3 px-4 font-semibold text-slate-800 whitespace-nowrap">
                                {item.model}
                              </td>
                              <td className="py-3 px-4 text-slate-600 max-w-xs truncate" title={item.question}>
                                {item.question}
                              </td>
                              <td className="py-3 px-4 text-right font-medium text-indigo-600 whitespace-nowrap">
                                {item.latency_seconds > 0 ? `${item.latency_seconds.toFixed(2)}s` : 'Err'}
                              </td>
                              <td className="py-3 px-4 text-right text-slate-500 whitespace-nowrap">
                                ~{item.token_count}
                              </td>
                              <td className="py-3 px-4">
                                <button
                                  onClick={() => setExpandedAnswerId(isExpanded ? null : item.id)}
                                  className="text-xs text-indigo-600 hover:text-indigo-800 font-medium inline-flex items-center gap-1"
                                >
                                  {isExpanded ? 'Hide' : 'View Answer'}
                                  {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                                </button>
                                {isExpanded && (
                                  <div className="mt-2 p-3 bg-slate-50 border border-slate-200 rounded-xl text-slate-700 whitespace-pre-wrap text-xs max-h-60 overflow-y-auto">
                                    {item.answer}
                                  </div>
                                )}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Historical Runs Dropdown / List */}
                {pastRuns.length > 1 && (
                  <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm space-y-3">
                    <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                      <Layers className="w-3.5 h-3.5" />
                      Past Benchmark Runs
                    </h3>
                    <div className="space-y-2">
                      {pastRuns.map((run, idx) => (
                        <button
                          key={run.run_group_id}
                          onClick={() => setCurrentRun(run)}
                          className={clsx(
                            "w-full flex items-center justify-between p-3 rounded-xl border text-xs transition-colors text-left",
                            activeRun?.run_group_id === run.run_group_id
                              ? "border-indigo-300 bg-indigo-50/50 text-indigo-950 font-medium"
                              : "border-slate-200 hover:bg-slate-50 text-slate-700"
                          )}
                        >
                          <div>
                            <span className="font-semibold">Run #{pastRuns.length - idx}</span>
                            <span className="text-slate-400 ml-2">({run.models.join(', ')})</span>
                          </div>
                          <div className="text-slate-400 text-right">
                            {new Date(run.created_at).toLocaleString()}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="bg-white border border-slate-200 rounded-2xl p-12 text-center text-slate-400 shadow-sm">
                <Gauge className="w-12 h-12 mx-auto mb-3 opacity-30 text-indigo-500" />
                <h3 className="text-base font-semibold text-slate-700 mb-1">No Benchmark Run Selected</h3>
                <p className="text-xs max-w-sm mx-auto">
                  Select Ollama models on the left and click &quot;Run Benchmark&quot; to test performance on your uploaded documents.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
