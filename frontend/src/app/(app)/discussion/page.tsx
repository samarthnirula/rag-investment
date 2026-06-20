"use client";

import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { listPosts, createPost, deletePost, Post } from "@/lib/api";

const POST_TYPES = ["Finding", "Question", "Note", "Risk", "Timeline", "Other"] as const;

const TYPE_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  Finding:  { bg: "bg-emerald-500/10 border-emerald-500/25", text: "text-emerald-300", dot: "bg-emerald-400" },
  Question: { bg: "bg-blue-500/10 border-blue-500/25",       text: "text-blue-300",    dot: "bg-blue-400"   },
  Note:     { bg: "bg-zinc-500/10 border-zinc-500/20",        text: "text-zinc-400",    dot: "bg-zinc-500"   },
  Risk:     { bg: "bg-red-500/10 border-red-500/25",          text: "text-red-300",     dot: "bg-red-400"    },
  Timeline: { bg: "bg-amber-500/10 border-amber-500/25",      text: "text-amber-300",   dot: "bg-amber-400"  },
  Other:    { bg: "bg-purple-500/10 border-purple-500/25",    text: "text-purple-300",  dot: "bg-purple-400" },
};

const TYPE_ICONS: Record<string, string> = {
  Finding: "✅", Question: "❓", Note: "📝", Risk: "🚩", Timeline: "📅", Other: "💬",
};

function Spinner() {
  return (
    <div className="flex gap-1.5">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

export default function DiscussionPage() {
  const { user, idToken, loading } = useAuth();
  const router = useRouter();

  const [posts, setPosts] = useState<Post[]>([]);
  const [postsLoading, setPostsLoading] = useState(false);
  const [filterType, setFilterType] = useState("All");
  const [showForm, setShowForm] = useState(false);
  const [postType, setPostType] = useState<string>("Note");
  const [content, setContent] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/");
  }, [user, loading, router]);

  const displayName = user?.displayName || user?.email || "User";

  const loadPosts = useCallback(async () => {
    if (!idToken) return;
    setPostsLoading(true);
    try {
      const data = await listPosts(idToken);
      setPosts(data);
    } catch {
      setError("Failed to load posts.");
    } finally {
      setPostsLoading(false);
    }
  }, [idToken]);

  useEffect(() => {
    if (idToken) loadPosts();
  }, [idToken, loadPosts]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!idToken || !content.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await createPost(idToken, { post_type: postType, content: content.trim() });
      setContent("");
      setShowForm(false);
      await loadPosts();
    } catch {
      setError("Failed to post. Please try again.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(postId: string) {
    if (!idToken) return;
    try {
      await deletePost(idToken, postId);
      setPosts((prev) => prev.filter((p) => p.post_id !== postId));
    } catch {
      setError("Failed to delete post.");
    }
  }

  function exportMarkdown() {
    const filtered = filterType === "All" ? posts : posts.filter((p) => p.type === filterType);
    let md = "# Discussion Board Export\n\n";
    for (const post of [...filtered].reverse()) {
      md += `## [${post.type}] ${post.author} — ${post.time}\n\n${post.content}\n\n---\n\n`;
    }
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "atticus_discussion.md";
    a.click();
    URL.revokeObjectURL(url);
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner />
      </div>
    );
  }

  if (!user) return null;

  const filtered = filterType === "All" ? posts : posts.filter((p) => p.type === filterType);
  const reversed = [...filtered].reverse();

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-5 border-b border-white/[0.07] shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-white">Discussion</h1>
            <p className="text-sm text-zinc-500 mt-0.5">
              Share findings, flag risks, and post questions with your team.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="text-xs bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-1.5 text-zinc-400 focus:outline-none focus:border-indigo-500/40"
            >
              <option value="All">All types</option>
              {POST_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <button
              onClick={loadPosts}
              className="text-xs text-zinc-500 hover:text-zinc-300 border border-white/[0.08] rounded-lg px-3 py-1.5 transition-colors"
            >
              ↺
            </button>
            {posts.length > 0 && (
              <button
                onClick={exportMarkdown}
                className="text-xs text-zinc-500 hover:text-zinc-300 border border-white/[0.08] rounded-lg px-3 py-1.5 transition-colors"
              >
                Export ↓
              </button>
            )}
            <button
              onClick={() => setShowForm((v) => !v)}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium rounded-lg px-4 py-1.5 transition-colors"
            >
              + Post
            </button>
          </div>
        </div>
      </div>

      {/* New post form */}
      {showForm && (
        <div className="px-6 py-4 border-b border-white/[0.07] bg-white/[0.02] shrink-0">
          <form onSubmit={handleSubmit} className="space-y-3 max-w-2xl">
            <div className="flex gap-2 flex-wrap">
              {POST_TYPES.map((t) => {
                const c = TYPE_COLORS[t] ?? TYPE_COLORS.Other;
                return (
                  <button
                    key={t}
                    type="button"
                    onClick={() => setPostType(t)}
                    className={`text-xs font-semibold px-3 py-1 rounded-full border transition-colors ${
                      postType === t
                        ? `${c.bg} border ${c.text}`
                        : "border-white/[0.08] text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    {TYPE_ICONS[t]} {t}
                  </button>
                );
              })}
            </div>
            <textarea
              autoFocus
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Share a finding, risk, date, or question from your documents…"
              rows={3}
              maxLength={4000}
              className="w-full bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50 resize-none"
            />
            <div className="flex items-center gap-2">
              <button
                type="submit"
                disabled={submitting || !content.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-5 py-2 transition-colors"
              >
                {submitting ? "Posting…" : "Post"}
              </button>
              <button
                type="button"
                onClick={() => { setShowForm(false); setContent(""); }}
                className="text-xs text-zinc-600 hover:text-zinc-400 border border-white/[0.07] rounded-lg px-4 py-2 transition-colors"
              >
                Cancel
              </button>
              <span className="text-[0.65rem] text-zinc-700 ml-auto">{content.length}/4000</span>
            </div>
          </form>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mx-6 mt-4 bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-2.5 text-xs text-red-300 shrink-0">
          {error}
          <button onClick={() => setError(null)} className="ml-2 text-red-400">×</button>
        </div>
      )}

      {/* Posts */}
      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-3">
        {postsLoading && posts.length === 0 && (
          <div className="flex justify-center py-12">
            <Spinner />
          </div>
        )}

        {!postsLoading && reversed.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16">
            <div className="text-3xl mb-3">💬</div>
            <p className="text-sm text-zinc-600">
              {filterType === "All"
                ? "No posts yet. Be the first to share a finding."
                : `No ${filterType} posts yet.`}
            </p>
          </div>
        )}

        {reversed.length > 0 && (
          <p className="text-[0.65rem] text-zinc-600">
            {reversed.length} post{reversed.length !== 1 ? "s" : ""}
            {filterType !== "All" && ` · filtered to ${filterType}`}
          </p>
        )}

        {reversed.map((post) => {
          const c = TYPE_COLORS[post.type] ?? TYPE_COLORS.Other;
          const isOwn = post.author === displayName;
          return (
            <div
              key={post.post_id}
              className={`flex items-start gap-3 bg-white/[0.02] border rounded-xl px-4 py-4 ${c.bg}`}
            >
              <div className={`w-2 h-2 rounded-full mt-1 shrink-0 ${c.dot}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                  <span className={`text-[0.65rem] font-bold uppercase tracking-wide ${c.text}`}>
                    {TYPE_ICONS[post.type]} {post.type}
                  </span>
                  <span className="text-xs font-semibold text-zinc-300">{post.author}</span>
                  <span className="text-[0.65rem] text-zinc-600">{post.time}</span>
                </div>
                <p className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap">
                  {post.content}
                </p>
              </div>
              {isOwn && (
                <button
                  onClick={() => handleDelete(post.post_id)}
                  className="text-zinc-700 hover:text-red-400 text-sm transition-colors shrink-0"
                  title="Delete post"
                >
                  ×
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
