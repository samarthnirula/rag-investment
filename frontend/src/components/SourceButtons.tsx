"use client";

import { useMemo, useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { SourceDetail } from "@/lib/api";

const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "how",
  "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
  "what", "when", "where", "which", "who", "why", "with",
  "case", "cases", "document", "documents", "epstein", "evidence", "file",
  "files", "issue", "issues", "key", "legal", "matter", "matters", "record",
  "records", "source", "sources", "summary",
]);

function queryTerms(query: string): string[] {
  const terms = query
    .toLowerCase()
    .match(/[a-z0-9][a-z0-9'-]{2,}/g) ?? [];
  return Array.from(new Set(terms.filter((term) => !STOP_WORDS.has(term))));
}

function bestSentence(text: string, terms: string[]): string {
  const sentences = text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence && sentence.split(/\s+/).length >= 8);

  if (sentences.length === 0) return text.trim();
  if (terms.length === 0) return sentences[0];

  const ranked = sentences
    .map((sentence) => {
      const lower = sentence.toLowerCase();
      const uniqueHits = terms.reduce((total, term) => total + (lower.includes(term) ? 1 : 0), 0);
      const exactHits = terms.reduce((total, term) => total + Math.min(lower.split(term).length - 1, 2), 0);
      const score = uniqueHits * 3 + exactHits;
      return { sentence, score, uniqueHits };
    })
    .sort((a, b) => b.score - a.score);

  if (ranked[0].score <= 0 || ranked[0].uniqueHits === 0) return sentences[0];
  return ranked[0].sentence;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Collapse all whitespace (including newlines) to single spaces, for comparing
 * a normalized sentence against a raw chunk that may have different line breaks. */
function normalizeWs(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

/** Build a regex source for `sentence` that matches it inside raw chunk text even
 * when the raw text has different whitespace/newlines than the normalized sentence
 * (bestSentence()/server-side excerpts both collapse whitespace before comparing). */
function sentencePatternSource(sentence: string): string {
  return escapeRegExp(sentence).replace(/ +/g, "\\s+");
}

function sourceTypeLabel(sourceType?: string): string {
  switch (sourceType) {
    case "case_overview":
      return "AI overview";
    case "case_timeline":
      return "AI timeline";
    case "demo_summary":
      return "Demo summary";
    case "public_context":
      return "Public context";
    case "user_note":
      return "User note";
    case "chat_memory":
      return "Chat memory";
    case "document":
    default:
      return "Document evidence";
  }
}

function sourceTypeClass(sourceType?: string): string {
  if (!sourceType || sourceType === "document") {
    return "border-emerald-400/20 bg-emerald-400/10 text-emerald-200";
  }
  return "border-amber-400/25 bg-amber-400/10 text-amber-200";
}

function sourceTitle(source: SourceDetail): string {
  return source.citation_label || source.label;
}

function HighlightedText({
  text,
  terms,
  sentence,
}: {
  text: string;
  terms: string[];
  sentence?: string;
}) {
  // Sentence pattern goes first in the alternation (and is whitespace-tolerant)
  // so a multi-line/extra-spaced sentence in the raw chunk text still matches
  // as one contiguous block, instead of falling through to per-word matches.
  const patternParts = [];
  const normalizedSentence = sentence ? normalizeWs(sentence) : "";
  if (normalizedSentence) patternParts.push(sentencePatternSource(normalizedSentence));
  patternParts.push(...terms.map(escapeRegExp));
  const pattern = patternParts.length > 0
    ? new RegExp(`(${patternParts.join("|")})`, "gi")
    : null;

  if (!pattern) return <>{text}</>;

  return (
    <>
      {text.split(pattern).map((part, index) => {
        if (!part) return null;
        if (normalizedSentence && normalizeWs(part) === normalizedSentence) {
          return (
            <mark key={index} className="rounded bg-amber-300/18 px-1 py-0.5 text-amber-100">
              {part}
            </mark>
          );
        }
        if (terms.some((term) => part.toLowerCase() === term.toLowerCase())) {
          return (
            <mark key={index} className="rounded bg-indigo-400/20 px-0.5 text-indigo-100">
              {part}
            </mark>
          );
        }
        return part;
      })}
    </>
  );
}

export function SourceButtons({
  sources,
  details,
  query,
  externalActive,
  onExternalChange,
  onSourceOpen,
}: {
  sources?: string[];
  details?: SourceDetail[];
  query?: string;
  /** When provided, the modal is controlled externally (e.g. by citation clicks). */
  externalActive?: SourceDetail | null;
  onExternalChange?: (d: SourceDetail | null) => void;
  /** Fired the first (and every) time a source pill is opened — handy for analytics/achievements. */
  onSourceOpen?: (source: SourceDetail) => void;
}) {
  const [internalActive, setInternalActive] = useState<SourceDetail | null>(null);

  const isControlled = externalActive !== undefined;
  const active       = isControlled ? externalActive : internalActive;
  const setActive    = isControlled
    ? (d: SourceDetail | null) => onExternalChange?.(d)
    : setInternalActive;

  // Escape key + click-outside to close
  const backdropRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!active) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setActive(null);
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [active]);
  const terms = useMemo(() => queryTerms(query ?? ""), [query]);
  const activeSentence = useMemo(
    () => (active ? (active.excerpt || bestSentence(active.chunk_text, terms)) : ""),
    [active, terms]
  );

  if ((!sources || sources.length === 0) && (!details || details.length === 0)) {
    return null;
  }

  return (
    <>
      <div className="mt-3 pt-3 border-t border-white/[0.07]">
        <p className="text-[0.65rem] font-bold text-zinc-600 tracking-[.08em] mb-2">SOURCES</p>
        <div className="flex flex-wrap gap-2">
          {(details && details.length > 0 ? details : []).map((source) => (
            <button
              key={`${source.index}-${source.chunk_id || source.document_id || source.label}-${source.page_number}`}
              type="button"
              onClick={() => { setActive(source); onSourceOpen?.(source); }}
              className="text-xs text-indigo-300 bg-indigo-500/12 border border-indigo-500/25 rounded-full px-2.5 py-1 transition-transform hover:scale-105 active:scale-95 hover:bg-indigo-500/18 hover:border-indigo-400/45 focus:outline-none focus:ring-2 focus:ring-indigo-400/40"
              title="Show retrieved evidence"
            >
              Source {source.index}: {sourceTitle(source)}
              <span className={`ml-1.5 rounded-full border px-1.5 py-0.5 text-[0.55rem] ${sourceTypeClass(source.source_type)}`}>
                {sourceTypeLabel(source.source_type)}
              </span>
              {source.jurisdiction && (
                <span className="ml-1.5 rounded-full border border-sky-400/20 bg-sky-400/10 px-1.5 py-0.5 text-[0.55rem] text-sky-200">
                  {source.jurisdiction}
                </span>
              )}
            </button>
          ))}
          {(!details || details.length === 0) && sources?.map((source) => (
            <span
              key={source}
              className="text-xs text-indigo-300 bg-indigo-500/12 border border-indigo-500/25 rounded-full px-2.5 py-1"
            >
              {source}
            </span>
          ))}
        </div>
      </div>

      <AnimatePresence>
        {active && (
          <motion.div
            ref={backdropRef}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-6"
            onClick={(e) => { if (e.target === backdropRef.current) setActive(null); }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.97, y: 4 }}
              transition={{ type: "spring", stiffness: 340, damping: 28 }}
              className="max-h-[86vh] w-full max-w-3xl overflow-hidden rounded-xl border border-white/[0.1] bg-[#101119] shadow-2xl"
            >
              <div className="flex items-start justify-between gap-4 border-b border-white/[0.08] px-5 py-4">
                <div>
                  <p className="text-xs font-bold text-indigo-300">Source {active.index}</p>
                  <h3 className="mt-1 text-sm font-semibold text-white">
                    {sourceTitle(active)}
                  </h3>
                  <p className="mt-1 text-xs text-zinc-500">
                    {active.file_name} · page {active.page_number}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className={`inline-flex rounded-full border px-2 py-0.5 text-[0.65rem] ${sourceTypeClass(active.source_type)}`}>
                      {sourceTypeLabel(active.source_type)}
                    </span>
                    {active.jurisdiction && (
                      <span className="inline-flex rounded-full border border-sky-400/20 bg-sky-400/10 px-2 py-0.5 text-[0.65rem] text-sky-200">
                        Jurisdiction: {active.jurisdiction}
                      </span>
                    )}
                  </div>
                  {active.section_header && (
                    <p className="mt-1 text-xs text-zinc-500">{active.section_header}</p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setActive(null)}
                  className="rounded-lg px-2 py-1 text-sm text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-zinc-200"
                >
                  Close
                </button>
              </div>

              <div className="max-h-[68vh] overflow-y-auto px-5 py-4">
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.05, duration: 0.2 }}
                  className="rounded-lg border border-amber-400/20 bg-amber-400/8 px-4 py-3"
                >
                  <p className="mb-2 text-[0.65rem] font-bold tracking-[.08em] text-amber-200/80">
                    BEST MATCHING PASSAGE
                  </p>
                  <p className="text-sm leading-relaxed text-amber-50">
                    <mark className="rounded bg-amber-300/18 px-1 py-0.5 text-amber-50">
                      {activeSentence}
                    </mark>
                  </p>
                </motion.div>

                <div className="mt-4 rounded-lg border border-white/[0.08] bg-white/[0.03] px-4 py-3">
                  <p className="mb-2 text-[0.65rem] font-bold tracking-[.08em] text-zinc-500">
                    FULL RETRIEVED CHUNK
                  </p>
                  {active.source_type && active.source_type !== "document" && (
                    <p className="mb-3 rounded-md border border-amber-400/20 bg-amber-400/8 px-3 py-2 text-xs leading-relaxed text-amber-100/80">
                      This source is generated or secondary context. Use it to orient research, then verify against primary documents before relying on it.
                    </p>
                  )}
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-300">
                    <HighlightedText text={active.chunk_text} terms={terms} sentence={activeSentence} />
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
