// lib/speech.ts — Web Speech API & Markdown Cleanup Utilities
//
// Uses browser-native Web Speech API (SpeechRecognition & SpeechSynthesis)
// for zero-dependency, offline voice input and text-to-speech output.

/**
 * Strips Markdown formatting from a text string so text-to-speech
 * reads cleanly without reciting formatting syntax like "asterisk asterisk".
 */
export function cleanMarkdownForSpeech(md: string): string {
  if (!md) return ''

  return md
    // Strip code blocks ```...```
    .replace(/```[\s\S]*?```/g, ' Code block omitted. ')
    // Strip inline code `...`
    .replace(/`([^`]+)`/g, '$1')
    // Strip images ![alt](url)
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    // Strip links [text](url) -> keep text
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    // Strip HTML tags
    .replace(/<[^>]*>/g, '')
    // Strip headers (# Header -> Header)
    .replace(/^#{1,6}\s+/gm, '')
    // Strip bold and italic (**text**, *text*, __text__, _text_)
    .replace(/(\*\*|__)(.*?)\1/g, '$2')
    .replace(/(\*|_)(.*?)\1/g, '$2')
    .replace(/~~(.*?)~~/g, '$1')
    // Strip blockquotes (> text)
    .replace(/^\s*>\s+/gm, '')
    // Strip list markers (- , * , 1. )
    .replace(/^\s*[\-\*\+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    // Strip horizontal rules (---, ***, ___)
    .replace(/^[\-\*_]{3,}\s*$/gm, '')
    // Replace multiple newlines with a single space / pause
    .replace(/\n+/g, ' ')
    // Normalize extra spaces
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * Check if text-to-speech (SpeechSynthesis) is supported in current browser.
 */
export function isTTSSupported(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window
}

/**
 * Check if speech recognition is supported in current browser.
 */
export function isSTTSupported(): boolean {
  if (typeof window === 'undefined') return false
  return !!((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)
}

/**
 * Speak text using SpeechSynthesis.
 * Returns the SpeechSynthesisUtterance object to allow cancellation or event binding.
 */
export function speakText(
  text: string,
  onEnd?: () => void,
  onError?: () => void,
): SpeechSynthesisUtterance | null {
  if (!isTTSSupported()) return null

  // Stop any currently ongoing speech first
  window.speechSynthesis.cancel()

  const clean = cleanMarkdownForSpeech(text)
  if (!clean) return null

  const utterance = new SpeechSynthesisUtterance(clean)
  utterance.rate = 1.0
  utterance.pitch = 1.0

  if (onEnd) utterance.onend = onEnd
  if (onError) utterance.onerror = onError

  window.speechSynthesis.speak(utterance)
  return utterance
}

/**
 * Stop any active text-to-speech output.
 */
export function stopSpeech(): void {
  if (isTTSSupported()) {
    window.speechSynthesis.cancel()
  }
}
