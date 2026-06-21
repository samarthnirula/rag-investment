"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { MarkdownMessage } from "@/components/MarkdownMessage";
import { SourceButtons } from "@/components/SourceButtons";
import { ImageModal } from "@/components/ImageModal";
import {
  listChats,
  createChat,
  deleteChat,
  renameChat,
  getMessages,
  saveMessage,
  runQueryStream,
  listCompanies,
  fetchImageObjectUrl,
  ChatSummary,
  Confidence,
  ImageAttachment,
  Message,
  SourceDetail,
} from "@/lib/api";

interface ActiveSource {
  msgIdx: number;
  detail: SourceDetail;
}

interface ActiveImage {
  src: string;
  alt: string;
  caption?: string;
}

function EvidenceImage({
  image,
  onOpen,
}: {
  image: ImageAttachment;
  onOpen: (src: string) => void;
}) {
  const [src, setSrc] = useState("");

  useEffect(() => {
    let active = true;
    let objectUrl = "";
    fetchImageObjectUrl(image.url)
      .then((url) => {
        objectUrl = url;
        if (active) setSrc(url);
        else URL.revokeObjectURL(url);
      })
      .catch(() => {});
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [image.url]);

  if (!src) {
    return <div className="h-32 w-full animate-pulse bg-white/[0.04]" />;
  }

  return (
    <button
      type="button"
      onClick={() => onOpen(src)}
      className="group overflow-hidden rounded-lg border border-white/[0.09] bg-white/[0.03] text-left transition-opacity hover:opacity-90"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={image.description || `Extracted image from page ${image.page_number}`}
        className="h-32 w-full object-contain bg-black/30 transition-transform group-hover:scale-[1.02]"
        loading="lazy"
      />
      <div className="px-2.5 py-2 text-[0.65rem] text-zinc-500">
        {image.source} · p.{image.page_number} · image {image.image_index + 1}
      </div>
    </button>
  );
}

interface WorkspaceSelection {
  raw: string;
  type: "epstein" | "case" | "general";
  caseId?: string;
  label: string;
}

function parseWorkspace(raw: string | null): WorkspaceSelection {
  if (!raw || raw === "epstein") {
    return { raw: "epstein", type: "epstein", label: "Epstein's case" };
  }
  if (raw.startsWith("case:")) {
    const [, caseId, ...labelParts] = raw.split(":");
    return {
      raw,
      type: "case",
      caseId,
      label: labelParts.join(":") || "Uploaded case",
    };
  }
  return { raw, type: "general", label: "All uploaded documents" };
}

function workspaceValueForChat(chat: ChatSummary): string {
  if (chat.case_id) return `case:${chat.case_id}:${chat.name}`;
  if (chat.page === "epstein") return "epstein";
  return "general";
}

const EXAMPLE_QUESTIONS = [
  "What are the key risk clauses in the contract?",
  "Summarize the defendant's testimony",
  "Find all mentions of wire transfers",
  "What precedents apply to this case?",
];

const FOLLOW_UP_SUGGESTIONS = [
  "Explain this in simpler terms",
  "What are the implications?",
  "Find related clauses",
  "Draft a response",
];

function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const score = Math.max(1, Math.min(5, confidence.score));
  const STOPS: [number, number, number][] = [
    [220, 38, 38], [234, 88, 12], [202, 138, 4], [101, 163, 13], [22, 163, 74],
  ];
  const lo = Math.min(Math.floor(score) - 1, 3);
  const hi = Math.min(lo + 1, 4);
  const f = score - Math.floor(score);
  const r = Math.round(STOPS[lo][0] + f * (STOPS[hi][0] - STOPS[lo][0]));
  const g = Math.round(STOPS[lo][1] + f * (STOPS[hi][1] - STOPS[lo][1]));
  const b = Math.round(STOPS[lo][2] + f * (STOPS[hi][2] - STOPS[lo][2]));
  const color = `rgb(${r},${g},${b})`;
  const LABELS: Record<number, string> = { 5: "High", 4: "Good", 3: "Moderate", 2: "Low", 1: "Unreliable" };
  const label = LABELS[score] ?? "Moderate";
  const ARC = Math.PI * 18;
  const filled = (score / 5) * ARC;
  return (
    <div
      className="mt-3 flex items-center gap-3 px-3 py-2.5 border rounded-lg"
      style={{ borderColor: `rgba(${r},${g},${b},0.3)`, backgroundColor: `rgba(${r},${g},${b},0.07)`, transition: "all 0.6s ease" }}
      title={confidence.rationale || undefined}
    >
      <svg width="48" height="28" viewBox="0 0 48 28" className="shrink-0" aria-hidden="true">
        <path d="M 6 24 A 18 18 0 0 1 42 24" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="4" strokeLinecap="round" />
        <path d="M 6 24 A 18 18 0 0 1 42 24" fill="none" stroke={color} strokeWidth="4" strokeLinecap="round"
          strokeDasharray={`${filled} ${ARC}`} style={{ transition: "stroke-dasharray 0.6s ease, stroke 0.6s ease" }} />
      </svg>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-1.5 flex-wrap">
          <span className="text-2xl font-bold leading-none" style={{ color, transition: "color 0.6s ease" }}>{score}</span>
          <span className="text-xs font-semibold" style={{ color, transition: "color 0.6s ease" }}>/5 · {label}</span>
        </div>
        {confidence.rationale && (
          <p className="text-xs text-zinc-500 mt-1 leading-relaxed">{confidence.rationale}</p>
        )}
      </div>
    </div>
  );
}

function RiskFlags({ content }: { content: string }) {
  const riskPattern = /⚠️?\s*(?:HIGH RISK|RISK|WARNING|CAUTION)[:\s]*.+/gi;
  const matches = content.match(riskPattern);
  const [expanded, setExpanded] = useState(false);
  if (!matches || matches.length === 0) return null;
  const visible = expanded ? matches : matches.slice(0, 3);
  const hasMore = matches.length > 3;

  return (
    <div className="mt-3 space-y-1.5">
      {visible.map((flag, i) => (
        <div
          key={i}
          className="flex items-start gap-2 px-2.5 py-2 border border-red-500/20 bg-red-500/[0.06] rounded-lg"
        >
          <span className="text-red-400 shrink-0">⚠️</span>
          <span className="text-xs text-red-300 leading-relaxed">{flag.replace(/^⚠️?\s*/, "")}</span>
        </div>
      ))}
      {hasMore && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors px-2.5"
        >
          {expanded ? "Show fewer" : `+${matches.length - 3} more risk flags`}
        </button>
      )}
    </div>
  );
}

function TrialExpiredModal() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-[#111218] border border-white/[0.1] rounded-2xl px-8 py-8 max-w-sm w-full mx-4 text-center shadow-2xl">
        <div className="text-4xl mb-4">⏰</div>
        <h2 className="text-lg font-bold text-white mb-2">Your 4-day trial has ended</h2>
        <p className="text-sm text-zinc-400 mb-6 leading-relaxed">
          Upgrade to continue asking questions and accessing your case files.
        </p>
        <a
          href="mailto:support@atticus.ai?subject=Upgrade%20my%20account"
          className="inline-block w-full bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm rounded-xl px-6 py-3 transition-colors"
        >
          Contact us to upgrade
        </a>
      </div>
    </div>
  );
}

export default function ChatPage() {
  const { user, loading, isTrialExpired } = useAuth();
  const router = useRouter();
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [querying, setQuerying] = useState(false);
  const [companies, setCompanies] = useState<string[]>([]);
  const [companyFilter, setCompanyFilter] = useState("");
  const [workspace, setWorkspace] = useState<WorkspaceSelection>(() => parseWorkspace(null));
  const [renamingChat, setRenamingChat] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [loadingChats, setLoadingChats] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTrialModal, setShowTrialModal] = useState(false);

  const [activeSource, setActiveSource] = useState<ActiveSource | null>(null);
  const [activeImage, setActiveImage] = useState<ActiveImage | null>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/");
  }, [user, loading, router]);

  useEffect(() => {
    const readWorkspace = () => {
      setWorkspace(parseWorkspace(localStorage.getItem("atticus_active_workspace")));
    };
    readWorkspace();
    const onWorkspaceChange = (event: Event) => {
      const custom = event as CustomEvent<string>;
      setWorkspace(parseWorkspace(custom.detail || localStorage.getItem("atticus_active_workspace")));
      if (activeChatId || messages.length > 0) {
        setActiveChatId(null);
        setMessages([]);
        setError("Workspace changed. Start a new chat for this case.");
      }
      setActiveSource(null);
    };
    window.addEventListener("atticus-workspace-change", onWorkspaceChange);
    window.addEventListener("storage", readWorkspace);
    return () => {
      window.removeEventListener("atticus-workspace-change", onWorkspaceChange);
      window.removeEventListener("storage", readWorkspace);
    };
  }, [activeChatId, messages.length]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const maxHeight = 5 * 24; // ~5 lines
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`;
  }, [input]);

  const loadChats = useCallback(async () => {
    if (!user) return;
    setLoadingChats(true);
    try {
      const list = await listChats();
      setChats(list.filter((c) => !c.page || c.page === "insightlens" || c.page === "epstein"));
    } catch {
      // non-critical
    } finally {
      setLoadingChats(false);
    }
  }, [user]);

  useEffect(() => {
    if (user) loadChats();
  }, [user, loadChats]);

  useEffect(() => {
    if (!user) return;
    listCompanies().then(setCompanies).catch(() => {});
  }, [user]);

  useEffect(() => {
    if (!user || !activeChatId) return;
    getMessages(activeChatId)
      .then(setMessages)
      .catch(() => setMessages([]));
    setActiveSource(null);
  }, [user, activeChatId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleNewChat() {
    if (!user) return;
    try {
      const { chat_id } = await createChat({
        page: workspace.type === "epstein" ? "epstein" : "insightlens",
        case_id: workspace.type === "case" ? workspace.caseId : undefined,
        name: "New Chat",
      });
      setActiveChatId(chat_id);
      setMessages([]);
      setActiveSource(null);
      await loadChats();
    } catch {
      setError("Failed to create chat.");
    }
  }

  function handleSelectChat(chat: ChatSummary) {
    const value = workspaceValueForChat(chat);
    localStorage.setItem("atticus_active_workspace", value);
    setWorkspace(parseWorkspace(value));
    window.dispatchEvent(new CustomEvent("atticus-workspace-sync", { detail: value }));
    setActiveChatId(chat.chat_id);
    setActiveSource(null);
    setError(null);
  }

  async function handleDeleteChat(chatId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!user) return;
    try {
      await deleteChat(chatId);
      if (activeChatId === chatId) { setActiveChatId(null); setMessages([]); setActiveSource(null); }
      await loadChats();
    } catch {
      setError("Failed to delete chat.");
    }
  }

  async function handleRenameSubmit(chatId: string) {
    if (!user || !renameValue.trim()) return;
    try {
      await renameChat(chatId, renameValue.trim());
      setRenamingChat(null);
      await loadChats();
    } catch {
      setError("Failed to rename chat.");
    }
  }

  async function submitQuery(text: string) {
    if (!text.trim() || querying || !user) return;
    if (isTrialExpired) { setShowTrialModal(true); return; }

    const inputText = text.trim();
    const userMsg: Message = { role: "user", content: inputText };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setQuerying(true);
    setError(null);
    setActiveSource(null);

    let placeholderAdded = false;

    try {
      let chatId = activeChatId;
      if (!chatId) {
        const { chat_id } = await createChat({
          name: inputText.slice(0, 40),
          page: workspace.type === "epstein" ? "epstein" : "insightlens",
          case_id: workspace.type === "case" ? workspace.caseId : undefined,
        });
        chatId = chat_id;
        setActiveChatId(chatId);
        await loadChats();
      } else {
        await saveMessage(chatId, { role: "user", content: inputText });
      }

      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
      placeholderAdded = true;

      let currentText = "";
      const currentChatId = chatId;

      const result = await runQueryStream(
        {
          query: inputText,
          chat_id: chatId,
          page: workspace.type === "epstein" ? "epstein" : "insightlens",
          case_id: workspace.type === "case" ? workspace.caseId : undefined,
          company_filter: workspace.type === "general" ? companyFilter || undefined : undefined,
        },
        (token) => {
          currentText += token;
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: currentText,
            };
            return updated;
          });
        },
      );

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: result.text,
          sources: result.sources,
          sourceDetails: result.source_details,
          images: result.images,
          imageNote: result.image_note,
          query: inputText,
          confidence: result.confidence,
        };
        return updated;
      });

      saveMessage(currentChatId, { role: "assistant", content: result.text }).catch(() => {});
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Query failed. Please try again.";
      if (msg === "TRIAL_EXPIRED") {
        setShowTrialModal(true);
      } else {
        setError(msg);
      }
      if (placeholderAdded) {
        setMessages((prev) => prev.slice(0, -1));
      }
    } finally {
      setQuerying(false);
    }
  }

  function handleSendQuery(e: React.FormEvent) {
    e.preventDefault();
    submitQuery(input);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuery(input);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <div key={i} className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
          ))}
        </div>
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="flex h-full">
      {(isTrialExpired || showTrialModal) && <TrialExpiredModal />}
      {/* ── Chat list sidebar ────────────────────────────── */}
      <div className="w-64 border-r border-white/[0.07] flex flex-col shrink-0">
        <div className="px-4 py-4 border-b border-white/[0.06]">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg px-4 py-2.5 transition-colors"
          >
            + New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto py-3 px-3 space-y-0.5">
          {loadingChats && chats.length === 0 && (
            <div className="text-center py-8">
              <div className="flex gap-1.5 justify-center mb-2">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
              <p className="text-xs text-zinc-600">Loading chats…</p>
            </div>
          )}

          {!loadingChats && chats.length === 0 && (
            <div className="text-center py-8 px-4">
              <p className="text-xs text-zinc-600 mb-1">No chats yet</p>
              <p className="text-xs text-zinc-700">Click + New Chat to start</p>
            </div>
          )}

          {chats.map((chat) => (
            <div key={chat.chat_id}>
              {renamingChat === chat.chat_id ? (
                <div className="px-2 py-1.5">
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onBlur={() => handleRenameSubmit(chat.chat_id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRenameSubmit(chat.chat_id);
                      if (e.key === "Escape") setRenamingChat(null);
                    }}
                    className="w-full bg-white/[0.06] border border-indigo-500/50 rounded px-2 py-1 text-xs text-white focus:outline-none"
                  />
                </div>
              ) : (
                <div
                  onClick={() => handleSelectChat(chat)}
                  className={`group flex items-center gap-2 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                    activeChatId === chat.chat_id
                      ? "bg-indigo-500/15 text-indigo-300"
                      : "hover:bg-white/[0.04] text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate">{chat.name}</div>
                    {chat.page && (
                      <div className="text-[0.6rem] text-zinc-600 truncate mt-0.5">
                        {chat.case_id ? "case-locked" : chat.page}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); setRenamingChat(chat.chat_id); setRenameValue(chat.name); }}
                    className="opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-zinc-400 transition-opacity text-xs"
                    title="Rename"
                  >
                    ✎
                  </button>
                  <button
                    onClick={(e) => handleDeleteChat(chat.chat_id, e)}
                    className="opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-red-400 transition-opacity text-xs"
                    title="Delete"
                  >
                    ×
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── Chat area ─────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-white/[0.07] shrink-0">
          <div className="text-sm font-medium text-zinc-400">
            {activeChatId
              ? chats.find((c) => c.chat_id === activeChatId)?.name || "Chat"
              : "Select or start a chat"}
            <span className="ml-2 rounded-full border border-indigo-400/20 bg-indigo-400/10 px-2 py-0.5 text-[0.65rem] text-indigo-200">
              {workspace.label}
            </span>
          </div>
          {workspace.type === "general" && companies.length > 0 && (
            <div className="flex items-center gap-2">
              <label className="text-xs text-zinc-600">Company:</label>
              <select
                value={companyFilter}
                onChange={(e) => setCompanyFilter(e.target.value)}
                className="text-xs bg-white/[0.05] border border-white/[0.1] rounded px-2 py-1 text-zinc-400 focus:outline-none focus:border-indigo-500/40"
              >
                <option value="">All companies</option>
                {companies.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
          {/* Empty state */}
          {!activeChatId && messages.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                {/* Scales illustration */}
                <div className="text-6xl mb-4">⚖️</div>
                <h2 className="text-lg font-semibold text-zinc-300 mb-2">Ask Atticus anything</h2>
                <p className="text-sm text-zinc-600 mb-6">
                  Upload case files and get AI-powered answers with exact page citations.
                </p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {EXAMPLE_QUESTIONS.map((q) => (
                    <button
                      key={q}
                      onClick={() => submitQuery(q)}
                      className="text-xs text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-3 py-1.5 hover:bg-indigo-500/20 transition-colors"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i}>
              {msg.role === "user" ? (
                <div className="flex justify-end">
                  <div className="bg-indigo-600/20 border border-indigo-500/30 rounded-tl-xl rounded-bl-xl rounded-br-sm px-4 py-3 max-w-xl text-sm text-zinc-200">
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div className="bg-white/[0.03] border border-white/[0.08] rounded-tr-xl rounded-br-xl rounded-bl-sm px-5 py-4 max-w-2xl">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-5 h-5 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500" />
                    <span className="text-xs font-semibold text-indigo-300">Atticus</span>
                  </div>

                  {/* Streaming cursor */}
                  {querying && i === messages.length - 1 && msg.content ? (
                    <div className="streaming-cursor">
                      <MarkdownMessage
                        content={msg.content}
                        onCitationClick={(n) => {
                          const detail = msg.sourceDetails?.find((d) => d.index === n);
                          if (detail) setActiveSource({ msgIdx: i, detail });
                        }}
                      />
                    </div>
                  ) : (
                    <MarkdownMessage
                      content={msg.content}
                      onCitationClick={(n) => {
                        const detail = msg.sourceDetails?.find((d) => d.index === n);
                        if (detail) setActiveSource({ msgIdx: i, detail });
                      }}
                    />
                  )}

                  {/* Confidence badge */}
                  {!querying && msg.confidence && (
                    <ConfidenceBadge confidence={msg.confidence} />
                  )}

                  {/* Risk flags */}
                  {!querying && msg.content && <RiskFlags content={msg.content} />}

                  {/* Extracted images */}
                  {msg.images && msg.images.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-white/[0.07]">
                      <p className="text-[0.65rem] font-bold text-zinc-600 tracking-[.08em] mb-2">
                        EXTRACTED IMAGES
                      </p>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                        {msg.images.map((image) => (
                          <EvidenceImage
                            key={image.image_id}
                            image={image}
                            onOpen={(src) =>
                              setActiveImage({
                                src,
                                alt: image.description || `Extracted image from page ${image.page_number}`,
                                caption: `${image.source} · p.${image.page_number} · image ${image.image_index + 1}`,
                              })
                            }
                          />
                        ))}
                      </div>
                    </div>
                  )}

                  {msg.imageNote && (
                    <div className="mt-4 rounded-lg border border-amber-500/20 bg-amber-500/8 px-3 py-2 text-xs leading-relaxed text-amber-200/90">
                      {msg.imageNote}
                    </div>
                  )}

                  <SourceButtons
                    sources={msg.sources}
                    details={msg.sourceDetails}
                    query={msg.query}
                    externalActive={activeSource?.msgIdx === i ? activeSource.detail : null}
                    onExternalChange={(d) => {
                      if (d) setActiveSource({ msgIdx: i, detail: d });
                      else setActiveSource(null);
                    }}
                  />

                  {/* Follow-up suggestion chips */}
                  {!querying && i === messages.length - 1 && msg.content && (
                    <div className="mt-4 pt-3 border-t border-white/[0.07] flex flex-wrap gap-2">
                      {FOLLOW_UP_SUGGESTIONS.map((s) => (
                        <button
                          key={s}
                          onClick={() => submitQuery(s)}
                          className="text-xs text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 rounded-full px-3 py-1.5 hover:bg-indigo-500/20 transition-colors"
                        >
                          {s} →
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Streaming / thinking indicator */}
          {querying && (messages.length === 0 || messages[messages.length - 1].role !== "assistant" || messages[messages.length - 1].content === "") && (
            <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl px-5 py-4 max-w-2xl">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-5 h-5 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500" />
                <span className="text-xs font-semibold text-indigo-300">Atticus</span>
              </div>
              <div className="flex gap-1.5">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-2.5 text-xs text-red-300">
              {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input bar — auto-resize textarea */}
        <div className="px-6 py-5 border-t border-white/[0.07] shrink-0">
          <form onSubmit={handleSendQuery} className="flex gap-3 max-w-3xl mx-auto items-end">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your case files…"
              disabled={querying}
              rows={1}
              className="flex-1 bg-white/[0.04] border border-white/[0.1] rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/25 disabled:opacity-50 transition-colors resize-none overflow-y-auto"
              style={{ maxHeight: `${5 * 24}px` }}
            />
            <button
              type="submit"
              disabled={querying || !input.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-medium text-sm rounded-xl px-5 py-3 transition-colors shrink-0"
            >
              {querying ? "…" : "Ask"}
            </button>
          </form>
        </div>
      </div>

      {/* ── Image lightbox ────────────────────────────────── */}
      {activeImage && (
        <ImageModal
          src={activeImage.src}
          alt={activeImage.alt}
          caption={activeImage.caption}
          onClose={() => setActiveImage(null)}
        />
      )}
    </div>
  );
}
