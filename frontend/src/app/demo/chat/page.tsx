"use client";

import { useState, useRef, useEffect, useCallback, useMemo, type InputHTMLAttributes } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  getDemoToken,
  getCachedDemoUserSlug,
  clearDemoToken,
  demoAssetUrl,
  demoMe,
  demoQuery,
  demoCases,
  demoUploadCase,
  demoTimeline,
  demoOverview,
  type DemoQueryResponse,
  type DemoCase,
  type DemoTimeline,
  type DemoOverview,
  type DemoTimelineEvent,
  type DemoSourceDetail,
} from "@/lib/demo-api";
import { MarkdownMessage } from "@/components/MarkdownMessage";
import { ImageModal } from "@/components/ImageModal";
import { SourceButtons } from "@/components/SourceButtons";
import { useAchievements, AchievementToastHost } from "@/components/Achievements";

const DEMO_HOURLY_LIMIT = 20; // mirrors backend/demo_router.py _DEMO_RATE_LIMIT, used only for the visual usage bar

// ── Types ──────────────────────────────────────────────────────────────────────

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  sourceDetails?: DemoSourceDetail[];
  confidence?: DemoQueryResponse["confidence"];
  cost_usd?: number;
}

type Tab = "chat" | "timeline" | "overview";

const folderPickerAttrs = {
  webkitdirectory: "",
  directory: "",
} as InputHTMLAttributes<HTMLInputElement>;

const ACTIVE_CASE_KEY_PREFIX = "atticus_demo_active_case:";
const CASES_CACHE_KEY_PREFIX = "atticus_demo_cases:";
const CHAT_CACHE_KEY_PREFIX = "atticus_demo_chat:";
const INTELLIGENCE_CACHE_KEY_PREFIX = "atticus_demo_intelligence:";
const MAX_CACHED_MESSAGES = 100;
const CASES_CACHE_TTL_MS = 5 * 60 * 1000;
const SHARED_INTELLIGENCE_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

function isSupportedCaseFile(file: File) {
  return /\.(pdf|pptx)$/i.test(file.name);
}

interface FileSystemFileEntryLike {
  isFile: true;
  isDirectory: false;
  file: (callback: (file: File) => void) => void;
  fullPath: string;
  name: string;
}

interface FileSystemDirectoryEntryLike {
  isFile: false;
  isDirectory: true;
  createReader: () => { readEntries: (callback: (entries: FileSystemEntryLike[]) => void) => void };
  fullPath: string;
  name: string;
}

type FileSystemEntryLike = FileSystemFileEntryLike | FileSystemDirectoryEntryLike;
type DataTransferItemWithEntry = DataTransferItem & {
  webkitGetAsEntry?: () => FileSystemEntryLike | null;
};

function activeCaseStorageKey(userSlug: string) {
  return `${ACTIVE_CASE_KEY_PREFIX}${userSlug}`;
}

function casesStorageKey(userSlug: string) {
  return `${CASES_CACHE_KEY_PREFIX}${userSlug}`;
}

function chatStorageKey(userSlug: string, caseId: string | null) {
  return `${CHAT_CACHE_KEY_PREFIX}${userSlug}:${caseId || "epstein"}`;
}

function readCachedCases(userSlug: string): DemoCase[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(casesStorageKey(userSlug)) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeCachedCases(userSlug: string, demoCases: DemoCase[]) {
  try {
    localStorage.setItem(casesStorageKey(userSlug), JSON.stringify(demoCases));
    localStorage.setItem(`${casesStorageKey(userSlug)}:cached_at`, String(Date.now()));
  } catch {
    // Storage may be unavailable or full; the live API remains the fallback.
  }
}

function hasFreshCasesCache(userSlug: string) {
  const cachedAt = Number(localStorage.getItem(`${casesStorageKey(userSlug)}:cached_at`) || 0);
  return cachedAt > 0 && Date.now() - cachedAt < CASES_CACHE_TTL_MS;
}

function readCachedMessages(userSlug: string, caseId: string | null): ChatMessage[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(chatStorageKey(userSlug, caseId)) || "[]");
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (message): message is ChatMessage =>
        message != null &&
        (message.role === "user" || message.role === "assistant") &&
        typeof message.content === "string",
    ).slice(-MAX_CACHED_MESSAGES);
  } catch {
    return [];
  }
}

function writeCachedMessages(userSlug: string, caseId: string | null, chatMessages: ChatMessage[]) {
  try {
    localStorage.setItem(
      chatStorageKey(userSlug, caseId),
      JSON.stringify(chatMessages.slice(-MAX_CACHED_MESSAGES)),
    );
  } catch {
    // Avoid breaking chat if browser storage is unavailable or full.
  }
}

function intelligenceStorageKey(
  userSlug: string,
  caseId: string | null,
  kind: "timeline" | "overview",
) {
  return `${INTELLIGENCE_CACHE_KEY_PREFIX}${userSlug}:${caseId || "epstein"}:${kind}`;
}

function readCachedIntelligence<T>(
  userSlug: string,
  caseId: string | null,
  kind: "timeline" | "overview",
): T | null {
  try {
    const raw = localStorage.getItem(intelligenceStorageKey(userSlug, caseId, kind));
    if (!raw) return null;
    const cached = JSON.parse(raw) as { value: T; cachedAt: number };
    const ttl = caseId ? Number.POSITIVE_INFINITY : SHARED_INTELLIGENCE_CACHE_TTL_MS;
    return Date.now() - cached.cachedAt <= ttl ? cached.value : null;
  } catch {
    return null;
  }
}

function writeCachedIntelligence<T>(
  userSlug: string,
  caseId: string | null,
  kind: "timeline" | "overview",
  value: T,
) {
  try {
    localStorage.setItem(
      intelligenceStorageKey(userSlug, caseId, kind),
      JSON.stringify({ value, cachedAt: Date.now() }),
    );
  } catch {
    // Live API remains available when browser storage is unavailable.
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function formatDate(iso: string) {
  try { return new Date(iso).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" }); }
  catch { return iso; }
}

// ── Components ─────────────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 px-1">
      <div className="flex gap-1.5">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
      <span className="text-[0.65rem] text-zinc-600">researching the case files…</span>
    </div>
  );
}

/** Gamified "energy bar" showing demo usage this session, with a color shift
 * as the user approaches the hourly limit — purely cosmetic feedback, the
 * real enforcement happens server-side in backend/demo_router.py. */
function UsageBar({ used, limit }: { used: number; limit: number }) {
  const pct = Math.min(100, (used / limit) * 100);
  const hot = pct >= 80;
  const warm = pct >= 50 && !hot;
  return (
    <div className="hidden sm:flex items-center gap-2" title={`${used} of ${limit} demo questions used this session`}>
      <span className="text-[0.6rem] text-zinc-600 whitespace-nowrap">session energy</span>
      <div className="w-20 h-1.5 rounded-full bg-white/[0.07] overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${
            hot ? "bg-gradient-to-r from-amber-400 to-rose-500" : warm ? "bg-gradient-to-r from-indigo-400 to-amber-400" : "bg-gradient-to-r from-indigo-500 to-purple-500"
          }`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 18 }}
        />
      </div>
    </div>
  );
}

function TimelineView({ data }: { data: DemoTimeline | null; loading: boolean }) {
  const [activeImage, setActiveImage] = useState<DemoTimelineEvent["image"] | null>(null);

  if (!data) return <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm">Loading timeline…</div>;
  if (data.pending) return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center text-zinc-500 text-sm">
        <div className="text-2xl mb-3">⏳</div>
        Timeline is being generated — check back shortly.
        {data.estimated_seconds && (
          <p className="mt-2 text-xs text-indigo-300">
            Estimated time: about {data.estimated_seconds} seconds
          </p>
        )}
        {data.note && <p className="text-xs text-zinc-600 mt-2">{data.note}</p>}
      </div>
    </div>
  );
  if (!data.events?.length) return <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm">No timeline events found.</div>;

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-end justify-between gap-4 mb-8">
          <div>
            <p className="text-[0.6rem] font-bold tracking-[.14em] text-indigo-400 mb-2">PUBLIC RECORD CHRONOLOGY</p>
            <h2 className="text-lg font-semibold text-zinc-100">Case Timeline</h2>
          </div>
          {data.generated_at && (
            <p className="text-[0.6rem] text-zinc-700">Generated {formatDate(data.generated_at)}</p>
          )}
        </div>

        <div className="relative pb-6">
          <div className="absolute bottom-0 top-0 left-4 w-px bg-gradient-to-b from-transparent via-white/[0.14] to-transparent sm:left-1/2" />

          <div className="space-y-8">
            {data.events.map((ev: DemoTimelineEvent, i: number) => {
              const isLeft = i % 2 === 0;
              const image = ev.image;
              const card = (
                <div className={`group rounded-lg border border-white/[0.08] bg-white/[0.03] p-4 shadow-lg shadow-black/10 transition-colors hover:border-indigo-400/30 ${isLeft ? "sm:text-right" : ""}`}>
                  {image && (
                    <button
                      type="button"
                      onClick={() => setActiveImage(image)}
                      className="mb-3 block w-full overflow-hidden rounded-md border border-white/[0.08] bg-black/20"
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={demoAssetUrl(image.url)}
                        alt={image.description || ev.title}
                        className="h-32 w-full object-cover opacity-85 transition duration-300 group-hover:scale-[1.02] group-hover:opacity-100"
                      />
                    </button>
                  )}
                  <div className="text-[0.65rem] text-indigo-400 font-mono mb-1">{ev.date}</div>
                  <div className="text-sm font-semibold text-zinc-100 mb-2">{ev.title}</div>
                  <p className="text-xs text-zinc-500 leading-relaxed">{ev.description}</p>
                  {(ev.source_doc || image) && (
                    <div className="mt-3 text-[0.6rem] text-zinc-700">
                      {ev.source_doc && <span>{ev.source_doc}{ev.page ? ` · p.${ev.page}` : ""}</span>}
                      {image && <span>{ev.source_doc ? " · " : ""}{image.source} · p.{image.page_number}</span>}
                    </div>
                  )}
                </div>
              );

              return (
                <div key={`${ev.date}-${ev.title}-${i}`} className="relative grid grid-cols-[2rem_1fr] gap-4 sm:grid-cols-[1fr_3.5rem_1fr] sm:gap-0">
                  <div className={`hidden sm:block ${isLeft ? "pr-6" : "sm:col-start-3 pl-6"}`}>
                    {card}
                  </div>

                  <div className="relative col-start-1 row-start-1 flex justify-center sm:col-start-2">
                    <div className="relative z-10 mt-5 flex h-7 w-7 items-center justify-center rounded-full border border-indigo-400/40 bg-[#08090e]">
                      <div className="h-2.5 w-2.5 rounded-full bg-indigo-400 shadow-[0_0_18px_rgba(129,140,248,0.65)]" />
                    </div>
                  </div>

                  <div className="col-start-2 sm:hidden">
                    {card}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {activeImage && (
          <ImageModal
            src={demoAssetUrl(activeImage.url)}
            alt={activeImage.description || "Timeline image"}
            caption={`${activeImage.source} · p.${activeImage.page_number}`}
            onClose={() => setActiveImage(null)}
          />
        )}
      </div>
    </div>
  );
}

function OverviewView({ data }: { data: DemoOverview | null; loading: boolean }) {
  if (!data) return <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm">Loading overview…</div>;
  if (data.pending) return (
    <div className="flex-1 flex items-center justify-center">
      <div className="text-center text-zinc-500 text-sm">
        <div className="text-2xl mb-3">⏳</div>
        Overview is being generated — check back shortly.
        {data.estimated_seconds && (
          <p className="mt-2 text-xs text-indigo-300">
            Estimated time: about {data.estimated_seconds} seconds
          </p>
        )}
        {data.note && <p className="text-xs text-zinc-600 mt-2">{data.note}</p>}
      </div>
    </div>
  );

  return (
    <div className="flex-1 overflow-y-auto px-6 py-6">
      <div className="max-w-2xl mx-auto space-y-5">
        <h2 className="text-sm font-semibold text-zinc-300">Case Overview</h2>

        {data.summary && (
          <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-5">
            <div className="text-[0.6rem] font-bold text-zinc-600 tracking-[.1em] mb-3">SUMMARY</div>
            <p className="text-sm text-zinc-300 leading-relaxed">{data.summary}</p>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4">
          {data.matter_type && (
            <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-4">
              <div className="text-[0.6rem] font-bold text-zinc-600 tracking-[.1em] mb-1.5">MATTER TYPE</div>
              <div className="text-sm text-zinc-300">{data.matter_type}</div>
            </div>
          )}
          {data.jurisdiction && (
            <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-4">
              <div className="text-[0.6rem] font-bold text-zinc-600 tracking-[.1em] mb-1.5">JURISDICTION</div>
              <div className="text-sm text-zinc-300">{data.jurisdiction}</div>
            </div>
          )}
        </div>

        {data.parties?.length > 0 && (
          <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-5">
            <div className="text-[0.6rem] font-bold text-zinc-600 tracking-[.1em] mb-3">KEY PARTIES</div>
            <div className="space-y-2">
              {data.parties.map((p, i) => (
                <div key={i} className="flex gap-3 items-start">
                  <span className="text-[0.6rem] text-indigo-400 bg-indigo-500/10 rounded px-1.5 py-0.5 font-mono whitespace-nowrap mt-0.5">
                    {p.role}
                  </span>
                  <span className="text-sm text-zinc-300">{p.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {data.key_issues?.length > 0 && (
          <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-5">
            <div className="text-[0.6rem] font-bold text-zinc-600 tracking-[.1em] mb-3">KEY ISSUES</div>
            <ul className="space-y-2">
              {data.key_issues.map((issue, i) => (
                <li key={i} className="flex gap-2 text-sm text-zinc-300">
                  <span className="text-indigo-400 mt-0.5">·</span>
                  {issue}
                </li>
              ))}
            </ul>
          </div>
        )}

        {data.generated_at && (
          <p className="text-[0.6rem] text-zinc-700 text-right">Generated {formatDate(data.generated_at)}</p>
        )}
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function DemoChatPage() {
  const router = useRouter();
  const [userSlug, setUserSlug]         = useState<string | null>(null);
  const [queryCount, setQueryCount]     = useState(0);
  const [unlimited, setUnlimited]       = useState(false);
  const [activeTab, setActiveTab]       = useState<Tab>("chat");
  const [messages, setMessages]         = useState<ChatMessage[]>([]);
  const [input, setInput]               = useState("");
  const [loading, setLoading]           = useState(false);
  const [timeline, setTimeline]         = useState<DemoTimeline | null>(null);
  const [tlLoading, setTlLoading]       = useState(false);
  const [overview, setOverview]         = useState<DemoOverview | null>(null);
  const [ovLoading, setOvLoading]       = useState(false);
  const [cases, setCases]               = useState<DemoCase[]>([]);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);
  const [showCaseModal, setShowCaseModal] = useState(false);
  const [caseName, setCaseName]         = useState("");
  const [caseFiles, setCaseFiles]       = useState<File[]>([]);
  const [caseUploading, setCaseUploading] = useState(false);
  const [caseUploadError, setCaseUploadError] = useState<string | null>(null);
  const caseFolderRef = useRef<HTMLInputElement>(null);
  const casePdfRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const messagesCacheKeyRef = useRef<string | null>(null);
  const { active: achievementToasts, unlock } = useAchievements();
  const visitedTabsRef = useRef<Set<Tab>>(new Set(["chat"]));

  const activateCase = useCallback((caseId: string | null, slug: string) => {
    setActiveCaseId(caseId);
    localStorage.setItem(activeCaseStorageKey(slug), caseId || "epstein");
    messagesCacheKeyRef.current = chatStorageKey(slug, caseId);
    setMessages(readCachedMessages(slug, caseId));
    setInput("");
  }, []);

  // Auth guard
  useEffect(() => {
    if (!getDemoToken()) { router.replace("/demo"); return; }
    const cachedSlug = getCachedDemoUserSlug();
    if (cachedSlug) {
      setUserSlug(cachedSlug);
      setUnlimited(cachedSlug === "user3");
      const cachedCases = readCachedCases(cachedSlug);
      setCases(cachedCases);
      const storedCaseId = localStorage.getItem(activeCaseStorageKey(cachedSlug));
      const cachedCaseId =
        storedCaseId && storedCaseId !== "epstein" &&
        cachedCases.some((demoCase) => demoCase.case_id === storedCaseId)
          ? storedCaseId
          : null;
      activateCase(cachedCaseId, cachedSlug);
    }
    demoMe()
      .then(async (me) => {
        setUserSlug(me.user_slug);
        setQueryCount(me.query_count);
        setUnlimited(Boolean(me.unlimited));
        const locallyCachedCases = readCachedCases(me.user_slug);
        if (locallyCachedCases.length > 0) setCases(locallyCachedCases);
        const cachedCaseId = localStorage.getItem(activeCaseStorageKey(me.user_slug));
        if (cachedCaseId === "epstein") {
          activateCase(null, me.user_slug);
        } else if (
          cachedCaseId &&
          locallyCachedCases.some((demoCase) => demoCase.case_id === cachedCaseId)
        ) {
          activateCase(cachedCaseId, me.user_slug);
        } else if (locallyCachedCases.length === 1) {
          activateCase(locallyCachedCases[0].case_id, me.user_slug);
        }

        // Reconcile quietly after rendering cached state, but not on every
        // refresh. Uploads update this cache immediately; the API refreshes it
        // at most once every five minutes.
        const loadedCases = hasFreshCasesCache(me.user_slug)
          ? null
          : await demoCases().catch(() => null);
        if (loadedCases) {
          setCases(loadedCases);
          writeCachedCases(me.user_slug, loadedCases);
          const selected = localStorage.getItem(activeCaseStorageKey(me.user_slug));
          if (
            selected &&
            selected !== "epstein" &&
            loadedCases.some((demoCase) => demoCase.case_id === selected)
          ) {
            activateCase(selected, me.user_slug);
          } else if (
            selected !== "epstein" &&
            selected &&
            !loadedCases.some((demoCase) => demoCase.case_id === selected)
          ) {
            const fallbackCaseId = loadedCases.length === 1 ? loadedCases[0].case_id : null;
            activateCase(fallbackCaseId, me.user_slug);
          } else if (!selected && loadedCases.length === 1) {
            activateCase(loadedCases[0].case_id, me.user_slug);
          }
        }
      })
      .catch(() => {
        // Never discard intake completion automatically. The cached session is
        // cleared only by explicit sign-out or browser storage controls.
      });
  }, [activateCase, router]);

  // Persist each case's conversation locally as it changes. The ref prevents
  // messages from one case being written under another case during a switch.
  useEffect(() => {
    if (!userSlug) return;
    const expectedKey = chatStorageKey(userSlug, activeCaseId);
    if (messagesCacheKeyRef.current !== expectedKey) return;
    writeCachedMessages(userSlug, activeCaseId, messages);
  }, [activeCaseId, messages, userSlug]);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Lazy-load timeline / overview when tab activated
  const loadTimeline = useCallback(async () => {
    if (timeline || tlLoading) return;
    if (userSlug) {
      const cached = readCachedIntelligence<DemoTimeline>(
        userSlug,
        activeCaseId,
        "timeline",
      );
      if (cached && !cached.pending) {
        setTimeline(cached);
        return;
      }
    }
    setTlLoading(true);
    try {
      const result = await demoTimeline(activeCaseId);
      setTimeline(result);
      if (userSlug && !result.pending) {
        writeCachedIntelligence(userSlug, activeCaseId, "timeline", result);
      }
    }
    catch (err) {
      setTimeline({
        pending: true,
        events: [],
        note: err instanceof Error ? err.message : "Failed to load timeline.",
      });
    }
    finally { setTlLoading(false); }
  }, [activeCaseId, timeline, tlLoading, userSlug]);

  const loadOverview = useCallback(async () => {
    if (overview || ovLoading) return;
    if (userSlug) {
      const cached = readCachedIntelligence<DemoOverview>(
        userSlug,
        activeCaseId,
        "overview",
      );
      if (cached && !cached.pending) {
        setOverview(cached);
        return;
      }
    }
    setOvLoading(true);
    try {
      const result = await demoOverview(activeCaseId);
      setOverview(result);
      if (userSlug && !result.pending) {
        writeCachedIntelligence(userSlug, activeCaseId, "overview", result);
      }
    }
    catch (err) {
      setOverview({
        pending: true,
        summary: "",
        parties: [],
        key_issues: [],
        note: err instanceof Error ? err.message : "Failed to load overview.",
      });
    }
    finally { setOvLoading(false); }
  }, [activeCaseId, overview, ovLoading, userSlug]);

  // Timeline and overview belong to the selected case. Never carry cached
  // Epstein data (or another uploaded case's data) across a case switch.
  useEffect(() => {
    setTimeline(null);
    setOverview(null);
    setTlLoading(false);
    setOvLoading(false);
  }, [activeCaseId]);

  // Also load from tab state so rapid clicks during initial auth/render cannot
  // leave a selected tab with no request in flight.
  useEffect(() => {
    if (activeTab === "timeline") void loadTimeline();
    if (activeTab === "overview") void loadOverview();
  }, [activeTab, loadTimeline, loadOverview]);

  // Uploaded-case intelligence may be generated asynchronously. Poll only
  // while a case-specific result is pending; shared Epstein data is static.
  useEffect(() => {
    if (
      !activeCaseId ||
      activeTab !== "timeline" ||
      !timeline?.pending ||
      !timeline.estimated_seconds
    ) return;
    const timer = window.setTimeout(() => setTimeline(null), 5000);
    return () => window.clearTimeout(timer);
  }, [activeCaseId, activeTab, timeline]);

  useEffect(() => {
    if (
      !activeCaseId ||
      activeTab !== "overview" ||
      !overview?.pending ||
      !overview.estimated_seconds
    ) return;
    const timer = window.setTimeout(() => setOverview(null), 5000);
    return () => window.clearTimeout(timer);
  }, [activeCaseId, activeTab, overview]);

  function switchTab(tab: Tab) {
    setActiveTab(tab);
    if (tab === "timeline") loadTimeline();
    if (tab === "overview") loadOverview();
    visitedTabsRef.current.add(tab);
    if (["chat", "timeline", "overview"].every((t) => visitedTabsRef.current.has(t as Tab))) {
      unlock("explorer");
    }
  }

  function startNewChat() {
    setActiveTab("chat");
    setMessages([]);
    setInput("");
    if (userSlug) writeCachedMessages(userSlug, activeCaseId, []);
  }

  function applyCaseFiles(files: File[]) {
    const selected = files.filter(isSupportedCaseFile);
    const folderName = (selected[0] as (File & { webkitRelativePath?: string }) | undefined)
      ?.webkitRelativePath
      ?.split("/")
      .filter(Boolean)[0];
    if (!caseName.trim() && folderName) setCaseName(folderName);
    setCaseFiles(selected);
    setCaseUploadError(selected.length === 0 ? "No PDF/PPTX files found. Choose a folder or files." : null);
  }

  function handleCaseFilesChange(e: React.ChangeEvent<HTMLInputElement>) {
    applyCaseFiles(Array.from(e.target.files ?? []));
  }

  function fileWithRelativePath(file: File, relativePath: string): File {
    try {
      Object.defineProperty(file, "webkitRelativePath", {
        value: relativePath.replace(/^\/+/, ""),
        configurable: true,
      });
    } catch {
      // Browser may expose webkitRelativePath as read-only; upload still works with file.name.
    }
    return file;
  }

  async function filesFromEntry(entry: FileSystemEntryLike, prefix = ""): Promise<File[]> {
    if (entry.isFile) {
      return new Promise((resolve) => {
        entry.file((file) => resolve([fileWithRelativePath(file, `${prefix}${file.name}`)]));
      });
    }

    const reader = entry.createReader();
    const entries = await new Promise<FileSystemEntryLike[]>((resolve) => reader.readEntries(resolve));
    const nested = await Promise.all(
      entries.map((child) => filesFromEntry(child, `${prefix}${entry.name}/`))
    );
    return nested.flat();
  }

  async function filesFromDrop(e: React.DragEvent<HTMLDivElement>): Promise<File[]> {
    const items = Array.from(e.dataTransfer.items ?? []) as DataTransferItemWithEntry[];
    const entryFiles = await Promise.all(
      items
        .map((item) => item.webkitGetAsEntry?.() as FileSystemEntryLike | null | undefined)
        .filter((entry): entry is FileSystemEntryLike => Boolean(entry))
        .map((entry) => filesFromEntry(entry))
    );
    const fromEntries = entryFiles.flat();
    if (fromEntries.length > 0) return fromEntries;
    return Array.from(e.dataTransfer.files ?? []);
  }

  async function handleCaseDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    if (caseUploading) return;
    applyCaseFiles(await filesFromDrop(e));
  }

  async function handleDemoCaseUpload() {
    if (caseUploading || caseFiles.length === 0) return;
    const name = caseName.trim() || "Uploaded Case";
    setCaseUploading(true);
    setCaseUploadError(null);
    try {
      const result = await demoUploadCase(caseFiles, name);
      const nextCases = await demoCases();
      setCases(nextCases);
      if (userSlug) {
        writeCachedCases(userSlug, nextCases);
        localStorage.removeItem(
          intelligenceStorageKey(userSlug, result.case_id, "timeline"),
        );
        localStorage.removeItem(
          intelligenceStorageKey(userSlug, result.case_id, "overview"),
        );
        activateCase(result.case_id, userSlug);
      } else {
        setActiveCaseId(result.case_id);
        setMessages([]);
      }
      setInput("");
      setActiveTab("chat");
      setShowCaseModal(false);
      setCaseName("");
      setCaseFiles([]);
      if (caseFolderRef.current) caseFolderRef.current.value = "";
      if (casePdfRef.current) casePdfRef.current.value = "";
    } catch (err) {
      setCaseUploadError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setCaseUploading(false);
    }
  }

  async function submitQuery(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", content: q }]);

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const res: DemoQueryResponse = await demoQuery(q, history, activeCaseId);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.answer,
          sources: res.sources,
          sourceDetails: res.source_details,
          confidence: res.confidence,
          cost_usd: res.cost_usd,
        },
      ]);
      setQueryCount((n) => {
        const next = n + 1;
        if (next === 1) unlock("first_question");
        if (next === 5) unlock("five_questions");
        if (next === 10) unlock("ten_questions");
        return next;
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      let display = "Something went wrong. Please try again.";
      if (msg.includes("rate limit") || msg.includes("RATE_LIMIT")) display = "Rate limit reached (20 queries/hour). Try again later.";
      if (msg.includes("401") || msg.includes("Invalid demo session")) {
        display = "The backend rejected this cached session. Please sign out and enter the demo code again.";
      }
      setMessages((prev) => [...prev, { role: "assistant", content: display }]);
    } finally {
      setLoading(false);
    }
  }

  function signOut() {
    clearDemoToken();
    router.push("/demo");
  }

  const TABS: { id: Tab; label: string }[] = [
    { id: "chat",     label: "Chat" },
    { id: "timeline", label: "Timeline" },
    { id: "overview", label: "Overview" },
  ];

  const STARTER_QUESTIONS = [
    "What are the key allegations in the Epstein case?",
    "Who are the main parties involved?",
    "What financial arrangements are described in the documents?",
    "Summarize the most significant court findings.",
  ];
  const activeCaseName = activeCaseId
    ? cases.find((demoCase) => demoCase.case_id === activeCaseId)?.case_name || "Uploaded case"
    : "Epstein Case";
  const hasUploadedDemoCase = cases.length > 0;

  return (
    <div className="flex h-screen bg-[#08090e] text-white">
      <AchievementToastHost achievements={achievementToasts} />
      <aside className="flex w-56 shrink-0 flex-col border-r border-white/[0.07] bg-[#08090e]">
        <div className="flex items-center gap-2 border-b border-white/[0.06] px-5 py-4">
          <div className="h-5 w-5 rounded bg-gradient-to-br from-indigo-500 to-purple-600" />
          <span className="text-sm font-bold tracking-tight text-zinc-100">Atticus</span>
        </div>

        <div className="flex gap-2 px-3 py-4">
          <button
            type="button"
            onClick={startNewChat}
            className="flex min-w-0 flex-1 items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500"
          >
            + New Chat
          </button>
          <button
            type="button"
            onClick={() => {
              if (hasUploadedDemoCase && !caseName.trim()) {
                setCaseName(cases[0]?.case_name || "Uploaded Case");
              }
              setShowCaseModal(true);
            }}
            title={hasUploadedDemoCase ? "Retry or add files to your demo case" : "Add a case"}
            aria-label="Add a case"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-white/[0.08] bg-white/[0.03] text-lg font-semibold text-zinc-300 transition-colors hover:border-indigo-400/30 hover:bg-white/[0.06]"
          >
            +
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2">
          <p className="mb-2 px-2 text-[0.65rem] font-bold uppercase tracking-[0.12em] text-zinc-700">Cases</p>
          <button
            type="button"
            onClick={() => {
              if (userSlug) activateCase(null, userSlug);
              setActiveTab("chat");
            }}
            className={`mb-1 flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs transition-colors ${
              activeCaseId === null
                ? "bg-indigo-500/15 text-indigo-300"
                : "text-zinc-500 hover:bg-white/[0.04] hover:text-zinc-300"
            }`}
          >
            <span className="truncate">Epstein&apos;s case</span>
            <span className="text-[0.6rem] text-zinc-700">demo</span>
          </button>
          {cases.map((demoCase) => (
            <button
              key={demoCase.case_id}
              type="button"
              onClick={() => {
                if (userSlug) activateCase(demoCase.case_id, userSlug);
                setActiveTab("chat");
              }}
              className={`mb-1 flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left text-xs transition-colors ${
                activeCaseId === demoCase.case_id
                  ? "bg-indigo-500/15 text-indigo-300"
                  : "text-zinc-500 hover:bg-white/[0.04] hover:text-zinc-300"
              }`}
            >
              <span className="truncate">{demoCase.case_name}</span>
              <span className="shrink-0 text-[0.6rem] text-zinc-700">{demoCase.document_count}</span>
            </button>
          ))}
          {hasUploadedDemoCase && !unlimited && (
            <p className="mt-3 rounded-lg border border-amber-400/15 bg-amber-400/5 px-3 py-2 text-[0.65rem] leading-relaxed text-amber-200/70">
              Demo upload limit reached: one full case per session.
            </p>
          )}
        </div>

        <div className="border-t border-white/[0.06] px-4 py-4">
          <button
            onClick={signOut}
            className="w-full rounded-lg border border-white/[0.07] px-3 py-1.5 text-xs text-zinc-600 transition-colors hover:text-zinc-400"
          >
            Sign out
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-white/[0.07] shrink-0">
        <div className="flex items-center gap-3">
          <div>
            <span className="text-sm font-bold text-zinc-100">{activeCaseName}</span>
            <span className="text-xs text-zinc-600 ml-2">— Demo Workspace</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {!unlimited && <UsageBar used={queryCount} limit={DEMO_HOURLY_LIMIT} />}
          {userSlug && (
            <span className="text-xs text-zinc-500 bg-white/[0.05] border border-white/[0.08] rounded-full px-2.5 py-1">
              {userSlug} · {queryCount} queries
            </span>
          )}
        </div>
      </header>

      {/* Tabs */}
      <div className="flex gap-1 px-6 pt-3 pb-0 border-b border-white/[0.07] shrink-0">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => switchTab(t.id)}
            className={`px-4 py-2 text-xs font-medium rounded-t-lg transition-colors border-b-2 -mb-px ${
              activeTab === t.id
                ? "text-indigo-300 border-indigo-500 bg-white/[0.03]"
                : "text-zinc-600 border-transparent hover:text-zinc-400"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === "chat" && (
        <>
          <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full">
                <div className="max-w-md w-full text-center space-y-4">
                  <div className="text-3xl">⚖️</div>
                  <p className="text-sm text-zinc-500">Ask anything about the Epstein case corpus.</p>
                  <div className="grid grid-cols-1 gap-2 text-left">
                    {STARTER_QUESTIONS.map((q, qi) => (
                      <motion.button
                        key={q}
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: qi * 0.05 }}
                        whileHover={{ scale: 1.015 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => { setInput(q); }}
                        className="text-xs text-zinc-400 bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.08] rounded-lg px-4 py-2.5 text-left transition-colors"
                      >
                        {q}
                      </motion.button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <AnimatePresence initial={false}>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                >
                  {msg.role === "user" ? (
                    <div className="flex justify-end">
                      <div className="bg-indigo-600/25 border border-indigo-500/25 rounded-xl px-4 py-2.5 max-w-xl text-sm text-zinc-200">
                        {msg.content}
                      </div>
                    </div>
                  ) : (
                    <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl px-5 py-4 max-w-2xl">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-4 h-4 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 shrink-0" />
                        <span className="text-xs font-semibold text-indigo-300">Atticus</span>
                        {msg.cost_usd != null && (
                          <span className="ml-auto text-[0.6rem] text-zinc-700">${msg.cost_usd.toFixed(4)}</span>
                        )}
                      </div>
                      <MarkdownMessage content={msg.content} />
                      {msg.confidence && (
                        <div
                          className={`mt-3 rounded-lg border px-3 py-2 ${
                            msg.confidence.score >= 4
                              ? "border-emerald-400/20 bg-emerald-400/8"
                              : msg.confidence.score >= 2.5
                              ? "border-amber-400/20 bg-amber-400/8"
                              : "border-rose-400/20 bg-rose-400/8"
                          }`}
                        >
                          <p
                            className={`text-[0.65rem] font-semibold ${
                              msg.confidence.score >= 4
                                ? "text-emerald-300"
                                : msg.confidence.score >= 2.5
                                ? "text-amber-300"
                                : "text-rose-300"
                            }`}
                          >
                            Confidence: {msg.confidence.rating} ({msg.confidence.score}/5)
                          </p>
                          {msg.confidence.rationale && (
                            <p className="mt-1 text-[0.65rem] leading-relaxed text-zinc-600">{msg.confidence.rationale}</p>
                          )}
                        </div>
                      )}
                      <SourceButtons
                        sources={msg.sources}
                        details={msg.sourceDetails}
                        query={messages[i - 1]?.role === "user" ? messages[i - 1].content : undefined}
                        onSourceOpen={() => unlock("source_diver")}
                      />
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {loading && (
              <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl px-5 py-4 max-w-2xl">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-4 h-4 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 shrink-0" />
                  <span className="text-xs font-semibold text-indigo-300">Atticus</span>
                </div>
                <TypingIndicator />
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="px-6 py-4 border-t border-white/[0.07] shrink-0">
            <form onSubmit={submitQuery} className="flex gap-3 max-w-3xl mx-auto">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about the case files…"
                disabled={loading}
                className="flex-1 bg-white/[0.04] border border-white/[0.1] rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/25 disabled:opacity-50 transition-colors"
              />
              <motion.button
                type="submit"
                disabled={loading || !input.trim()}
                whileHover={!loading && input.trim() ? { scale: 1.04 } : undefined}
                whileTap={!loading && input.trim() ? { scale: 0.96 } : undefined}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-semibold text-sm rounded-xl px-5 py-3 transition-colors shrink-0"
              >
                Ask
              </motion.button>
            </form>
            <p className="text-[0.6rem] text-zinc-700 text-center mt-2">
              AI responses are research aids only — not legal advice.
            </p>
          </div>
        </>
      )}

      <AnimatePresence mode="wait">
        {activeTab === "timeline" && (
          <motion.div
            key="timeline"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="flex-1 overflow-hidden flex flex-col"
          >
            <TimelineView data={timeline} loading={tlLoading} />
          </motion.div>
        )}

        {activeTab === "overview" && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="flex-1 overflow-hidden flex flex-col"
          >
            <OverviewView data={overview} loading={ovLoading} />
          </motion.div>
        )}
      </AnimatePresence>
      </div>

      {showCaseModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-6">
          <div className="w-full max-w-lg rounded-2xl border border-white/[0.1] bg-[#101119] shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/[0.08] px-5 py-4">
              <div>
                <h2 className="text-sm font-semibold text-white">Add a demo case</h2>
                <p className="mt-1 text-xs text-zinc-600">Upload a folder of PDFs/PPTXs to test Atticus on your own matter.</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  if (caseUploading) return;
                  setShowCaseModal(false);
                  setCaseUploadError(null);
                }}
                className="rounded-lg px-2 py-1 text-sm text-zinc-500 transition-colors hover:bg-white/[0.06] hover:text-zinc-200"
              >
                Close
              </button>
            </div>

            <div className="space-y-4 px-5 py-5">
              <div>
                <label className="mb-1 block text-xs text-zinc-500">Case name</label>
                <input
                  value={caseName}
                  onChange={(e) => setCaseName(e.target.value)}
                  placeholder="Auto-filled from folder name"
                  disabled={caseUploading}
                  className="w-full rounded-lg border border-white/[0.1] bg-white/[0.05] px-3 py-2 text-sm text-white placeholder-zinc-600 outline-none transition-colors focus:border-indigo-500/50 disabled:opacity-50"
                />
              </div>

              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleCaseDrop}
                className="flex flex-col items-center gap-3 rounded-xl border-2 border-dashed border-white/[0.12] px-5 py-6 text-center transition-colors hover:border-indigo-500/40"
              >
                <span className="text-2xl">+</span>
                <span className="text-sm text-zinc-400">
                  {caseFiles.length > 0
                    ? `${caseFiles.length} file${caseFiles.length === 1 ? "" : "s"} selected`
                    : "Drop a folder or PDF/PPTX files here"}
                </span>
                <span className="text-xs text-zinc-600">
                  {unlimited ? "No upload limits for this demo account" : "Demo limit: 8 files, 25 MB each"}
                </span>
                <div className="flex flex-wrap justify-center gap-2">
                  <button
                    type="button"
                    onClick={() => caseFolderRef.current?.click()}
                    disabled={caseUploading}
                    className="rounded-lg border border-indigo-400/25 bg-indigo-400/10 px-3 py-2 text-xs font-medium text-indigo-200 transition-colors hover:bg-indigo-400/15 disabled:opacity-50"
                  >
                    Choose folder
                  </button>
                  <button
                    type="button"
                    onClick={() => casePdfRef.current?.click()}
                    disabled={caseUploading}
                    className="rounded-lg border border-white/[0.08] bg-white/[0.04] px-3 py-2 text-xs font-medium text-zinc-300 transition-colors hover:bg-white/[0.07] disabled:opacity-50"
                  >
                    Choose files
                  </button>
                </div>
                <input
                  {...folderPickerAttrs}
                  ref={caseFolderRef}
                  type="file"
                  accept=".pdf,.pptx"
                  multiple
                  disabled={caseUploading}
                  className="hidden"
                  onChange={handleCaseFilesChange}
                />
                <input
                  ref={casePdfRef}
                  type="file"
                  accept=".pdf,.pptx"
                  multiple
                  disabled={caseUploading}
                  className="hidden"
                  onChange={handleCaseFilesChange}
                />
              </div>

              {caseFiles.length > 0 && (
                <div className="max-h-28 overflow-y-auto rounded-lg border border-white/[0.07] bg-white/[0.03] px-3 py-2">
                  {caseFiles.slice(0, 8).map((file) => {
                    const path = (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
                    return (
                      <p key={path} className="truncate text-[0.65rem] leading-5 text-zinc-500">
                        {path}
                      </p>
                    );
                  })}
                </div>
              )}

              {caseUploadError && (
                <div className="rounded-lg border border-red-500/25 bg-red-500/10 px-4 py-2.5 text-xs text-red-300">
                  {caseUploadError}
                </div>
              )}

              <button
                type="button"
                onClick={handleDemoCaseUpload}
                disabled={caseUploading || caseFiles.length === 0}
                className="w-full rounded-lg bg-indigo-600 py-2.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {caseUploading
                  ? "Processing case… usually 15–60 seconds"
                  : "Upload case"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
