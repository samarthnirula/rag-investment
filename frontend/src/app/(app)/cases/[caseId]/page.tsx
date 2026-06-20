"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import { MarkdownMessage } from "@/components/MarkdownMessage";
import { SourceButtons } from "@/components/SourceButtons";
import {
  getCaseChats,
  getCaseOverview,
  getCaseTimeline,
  runQueryStream,
  getMessages,
  saveMessage,
  CaseChat,
  CaseOverview,
  CaseTimeline,
  Message,
  SourceDetail,
} from "@/lib/api";
import { CASE_TAB_TEMPLATE, type CaseTab } from "@/lib/caseTemplate";

interface ActiveSource {
  msgIdx: number;
  detail: SourceDetail;
}

function TrialExpiredModal() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-[#111218] border border-white/[0.1] rounded-2xl px-8 py-8 max-w-sm w-full mx-4 text-center shadow-2xl">
        <div className="text-4xl mb-4">⏰</div>
        <h2 className="text-lg font-bold text-white mb-2">Your 4-day trial has ended</h2>
        <p className="text-sm text-zinc-400 mb-6 leading-relaxed">
          Upgrade to continue chatting and accessing your case files.
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

function Spinner() {
  return (
    <div className="flex gap-1.5 justify-center">
      {[0, 1, 2].map((i) => (
        <div key={i} className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
      ))}
    </div>
  );
}

export default function CaseDetailPage() {
  const params = useParams();
  const caseId = params.caseId as string;
  const router = useRouter();
  const { user, loading, isTrialExpired } = useAuth();

  const [activeTab, setActiveTab] = useState<CaseTab["id"]>("chat");
  const [caseChats, setCaseChats] = useState<CaseChat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [querying, setQuerying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showTrialModal, setShowTrialModal] = useState(false);
  const [activeSource, setActiveSource] = useState<ActiveSource | null>(null);

  const [overview, setOverview] = useState<CaseOverview | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(false);

  const [timeline, setTimeline] = useState<CaseTimeline | null>(null);
  const [timelineLoading, setTimelineLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/");
  }, [user, loading, router]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 5 * 24)}px`;
  }, [input]);

  const loadChats = useCallback(async () => {
    if (!caseId) return;
    try {
      const chats = await getCaseChats(caseId);
      setCaseChats(chats);
      if (!activeChatId) {
        const chatTab = chats.find((c) => c.chat_type === "chat");
        if (chatTab) setActiveChatId(chatTab.chat_id);
      }
    } catch {
      // non-critical
    }
  }, [caseId, activeChatId]);

  useEffect(() => {
    if (user && caseId) loadChats();
  }, [user, caseId, loadChats]);

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

  useEffect(() => {
    if (activeTab === "overview" && !overview && !overviewLoading) {
      setOverviewLoading(true);
      getCaseOverview(caseId)
        .then(setOverview)
        .catch(() => {})
        .finally(() => setOverviewLoading(false));
    }
  }, [activeTab, overview, overviewLoading, caseId]);

  useEffect(() => {
    if (activeTab === "timeline" && !timeline && !timelineLoading) {
      setTimelineLoading(true);
      getCaseTimeline(caseId)
        .then(setTimeline)
        .catch(() => {})
        .finally(() => setTimelineLoading(false));
    }
  }, [activeTab, timeline, timelineLoading, caseId]);

  async function submitQuery(text: string) {
    if (!text.trim() || querying || !user || !activeChatId) return;
    if (isTrialExpired) { setShowTrialModal(true); return; }

    const inputText = text.trim();
    setMessages((prev) => [...prev, { role: "user", content: inputText }]);
    setInput("");
    setQuerying(true);
    setError(null);
    setActiveSource(null);

    let placeholderAdded = false;
    try {
      await saveMessage(activeChatId, { role: "user", content: inputText });
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
      placeholderAdded = true;

      let currentText = "";
      const result = await runQueryStream(
        { query: inputText, chat_id: activeChatId, page: "insightlens", case_id: caseId },
        (token) => {
          currentText += token;
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...updated[updated.length - 1], content: currentText };
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
          query: inputText,
        };
        return updated;
      });

      saveMessage(activeChatId, { role: "assistant", content: result.text }).catch(() => {});
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Query failed.";
      if (msg === "TRIAL_EXPIRED") {
        setShowTrialModal(true);
      } else {
        setError(msg);
      }
      if (placeholderAdded) setMessages((prev) => prev.slice(0, -1));
    } finally {
      setQuerying(false);
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner />
      </div>
    );
  }

  if (!user) return null;

  return (
    <div className="flex flex-col h-full">
      {(isTrialExpired || showTrialModal) && <TrialExpiredModal />}

      {/* Tab bar */}
      <div className="flex items-center gap-1 px-6 pt-4 border-b border-white/[0.07] shrink-0">
        <button
          onClick={() => router.push("/cases")}
          className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors mr-3"
        >
          ← Cases
        </button>
        {CASE_TAB_TEMPLATE.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
              activeTab === tab.id
                ? "border-indigo-500 text-indigo-300"
                : "border-transparent text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* Chat tab */}
      {activeTab === "chat" && (
        <div className="flex flex-1 min-h-0">
          {/* Chat selector sidebar */}
          <div className="w-56 border-r border-white/[0.07] flex flex-col shrink-0">
            <div className="px-3 py-3 border-b border-white/[0.06]">
              <p className="text-xs text-zinc-600 font-medium px-1">Case Chats</p>
            </div>
            <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
              {caseChats
                .filter((c) => c.chat_type === "chat")
                .map((chat) => (
                  <div
                    key={chat.chat_id}
                    onClick={() => setActiveChatId(chat.chat_id)}
                    className={`px-3 py-2 rounded-lg cursor-pointer text-xs transition-colors ${
                      activeChatId === chat.chat_id
                        ? "bg-indigo-500/15 text-indigo-300 font-medium"
                        : "text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04]"
                    }`}
                  >
                    💬 {chat.name}
                  </div>
                ))}
            </div>
          </div>

          {/* Chat area */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 overflow-y-auto px-6 py-5 space-y-4">
              {!activeChatId && (
                <div className="flex items-center justify-center h-full text-center">
                  <div>
                    <div className="text-4xl mb-3">💬</div>
                    <p className="text-sm text-zinc-500">Select a chat to begin</p>
                  </div>
                </div>
              )}

              {activeChatId && messages.length === 0 && !querying && (
                <div className="flex items-center justify-center h-full text-center">
                  <div>
                    <div className="text-4xl mb-3">⚖️</div>
                    <p className="text-sm font-medium text-zinc-400 mb-1">Ask about this case</p>
                    <p className="text-xs text-zinc-600">Responses are scoped to documents in this case.</p>
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
                      <MarkdownMessage
                        content={msg.content}
                        onCitationClick={(n) => {
                          const detail = msg.sourceDetails?.find((d) => d.index === n);
                          if (detail) setActiveSource({ msgIdx: i, detail });
                        }}
                      />
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
                    </div>
                  )}
                </div>
              ))}

              {querying && (
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

            {/* Input */}
            <div className="px-6 py-4 border-t border-white/[0.07] shrink-0">
              <form onSubmit={(e) => { e.preventDefault(); submitQuery(input); }} className="flex gap-3">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitQuery(input); }
                  }}
                  placeholder="Ask about documents in this case…"
                  disabled={querying || !activeChatId}
                  rows={1}
                  className="flex-1 bg-white/[0.04] border border-white/[0.09] rounded-xl px-4 py-3 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-indigo-500/40 resize-none transition-colors disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!input.trim() || querying || !activeChatId}
                  className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-sm font-medium rounded-xl px-5 py-3 transition-colors shrink-0"
                >
                  {querying ? "…" : "Ask"}
                </button>
              </form>
            </div>
          </div>
        </div>
      )}

      {/* Timeline tab */}
      {activeTab === "timeline" && (
        <div className="flex-1 overflow-y-auto px-8 py-6">
          {timelineLoading && (
            <div className="flex justify-center py-12"><Spinner /></div>
          )}
          {!timelineLoading && timeline?.pending && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="text-4xl mb-3">📅</div>
              <p className="text-sm font-medium text-zinc-400 mb-1">Generating timeline…</p>
              <p className="text-xs text-zinc-600">This happens automatically after documents are ingested.</p>
            </div>
          )}
          {!timelineLoading && timeline && !timeline.pending && (
            <div className="max-w-2xl mx-auto">
              <h2 className="text-base font-semibold text-zinc-300 mb-6">Case Timeline</h2>
              <div className="relative border-l border-white/[0.08] pl-6 space-y-6">
                {(timeline.events ?? []).map((event, i) => (
                  <div key={i} className="relative">
                    <div className="absolute -left-[25px] w-3 h-3 rounded-full bg-indigo-500/60 border border-indigo-500 mt-1" />
                    <p className="text-[0.65rem] text-indigo-400 font-mono mb-1">{event.date}</p>
                    <p className="text-sm font-semibold text-zinc-200 mb-0.5">{event.title}</p>
                    <p className="text-xs text-zinc-500 leading-relaxed">{event.description}</p>
                    {event.source_doc && (
                      <p className="text-[0.6rem] text-zinc-700 mt-1">{event.source_doc}{event.page ? ` · p.${event.page}` : ""}</p>
                    )}
                  </div>
                ))}
                {(timeline.events ?? []).length === 0 && (
                  <p className="text-xs text-zinc-600">No timeline events were extracted.</p>
                )}
              </div>
            </div>
          )}
          {!timelineLoading && !timeline && (
            <div className="flex justify-center py-12">
              <p className="text-xs text-zinc-600">Could not load timeline.</p>
            </div>
          )}
        </div>
      )}

      {/* Overview tab */}
      {activeTab === "overview" && (
        <div className="flex-1 overflow-y-auto px-8 py-6">
          {overviewLoading && (
            <div className="flex justify-center py-12"><Spinner /></div>
          )}
          {!overviewLoading && overview?.pending && (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="text-4xl mb-3">📋</div>
              <p className="text-sm font-medium text-zinc-400 mb-1">Generating overview…</p>
              <p className="text-xs text-zinc-600">This happens automatically after documents are ingested.</p>
            </div>
          )}
          {!overviewLoading && overview && !overview.pending && (
            <div className="max-w-2xl mx-auto space-y-6">
              <h2 className="text-base font-semibold text-zinc-300">Case Overview</h2>

              {overview.summary && (
                <div className="bg-white/[0.03] border border-white/[0.07] rounded-xl px-5 py-4">
                  <p className="text-[0.65rem] font-bold text-zinc-600 tracking-[.08em] mb-2">SUMMARY</p>
                  <p className="text-sm text-zinc-300 leading-relaxed">{overview.summary}</p>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4">
                {overview.matter_type && (
                  <div className="bg-white/[0.03] border border-white/[0.07] rounded-xl px-4 py-3">
                    <p className="text-[0.6rem] font-bold text-zinc-600 tracking-[.08em] mb-1">MATTER TYPE</p>
                    <p className="text-xs text-zinc-300">{overview.matter_type}</p>
                  </div>
                )}
                {overview.jurisdiction && (
                  <div className="bg-white/[0.03] border border-white/[0.07] rounded-xl px-4 py-3">
                    <p className="text-[0.6rem] font-bold text-zinc-600 tracking-[.08em] mb-1">JURISDICTION</p>
                    <p className="text-xs text-zinc-300">{overview.jurisdiction}</p>
                  </div>
                )}
              </div>

              {(overview.parties ?? []).length > 0 && (
                <div className="bg-white/[0.03] border border-white/[0.07] rounded-xl px-5 py-4">
                  <p className="text-[0.65rem] font-bold text-zinc-600 tracking-[.08em] mb-3">PARTIES</p>
                  <div className="space-y-1.5">
                    {(overview.parties ?? []).map((p, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <span className="text-[0.6rem] text-indigo-400 font-semibold w-20 shrink-0">{p.role}</span>
                        <span className="text-xs text-zinc-300">{p.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(overview.key_issues ?? []).length > 0 && (
                <div className="bg-white/[0.03] border border-white/[0.07] rounded-xl px-5 py-4">
                  <p className="text-[0.65rem] font-bold text-zinc-600 tracking-[.08em] mb-3">KEY ISSUES</p>
                  <ul className="space-y-1.5">
                    {(overview.key_issues ?? []).map((issue, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-zinc-300">
                        <span className="text-indigo-400 shrink-0 mt-0.5">•</span>
                        {issue}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          {!overviewLoading && !overview && (
            <div className="flex justify-center py-12">
              <p className="text-xs text-zinc-600">Could not load overview.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
