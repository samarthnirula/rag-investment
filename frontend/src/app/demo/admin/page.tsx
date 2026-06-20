"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { adminCosts, type AdminCosts } from "@/lib/demo-api";

const ADMIN_KEY_STORAGE = "demo_admin_key";

function fmt(n: number) { return `$${n.toFixed(4)}`; }
function fmtDate(iso: string | null) {
  if (!iso) return "—";
  try { return new Date(iso).toLocaleString(); }
  catch { return iso; }
}

export default function DemoAdminPage() {
  const [adminKey, setAdminKey] = useState<string>(() =>
    typeof window !== "undefined" ? sessionStorage.getItem(ADMIN_KEY_STORAGE) || "" : ""
  );
  const [keyInput, setKeyInput]   = useState("");
  const [authed, setAuthed]       = useState(false);
  const [data, setData]           = useState<AdminCosts | null>(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadData = useCallback(async (key: string) => {
    setLoading(true);
    setError("");
    try {
      const result = await adminCosts(key);
      setData(result);
      setAuthed(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      if (msg.includes("403") || msg.includes("Forbidden") || msg.includes("Invalid")) {
        setError("Invalid admin key.");
        setAuthed(false);
        sessionStorage.removeItem(ADMIN_KEY_STORAGE);
      } else {
        setError(`Load failed: ${msg}`);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-refresh every 30 seconds when authenticated
  useEffect(() => {
    if (!authed || !adminKey) return;
    intervalRef.current = setInterval(() => loadData(adminKey), 30_000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [authed, adminKey, loadData]);

  // If we had a stored key, try it immediately
  useEffect(() => {
    if (adminKey) loadData(adminKey);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleKeySubmit(e: React.FormEvent) {
    e.preventDefault();
    const key = keyInput.trim();
    if (!key) return;
    sessionStorage.setItem(ADMIN_KEY_STORAGE, key);
    setAdminKey(key);
    loadData(key);
  }

  if (!authed) {
    return (
      <div className="min-h-screen bg-[#08090e] flex items-center justify-center px-4">
        <div className="w-full max-w-sm">
          <div className="flex items-center justify-center gap-3 mb-8">
            <div className="w-7 h-7 rounded-md bg-gradient-to-br from-indigo-500 to-purple-600" />
            <span className="text-xl font-bold text-white tracking-tight">ATTICUS</span>
            <span className="text-xs text-zinc-500 bg-white/[0.05] border border-white/[0.08] rounded px-2 py-0.5">Admin</span>
          </div>
          <form
            onSubmit={handleKeySubmit}
            className="bg-white/[0.03] border border-white/[0.09] rounded-2xl px-8 py-7 space-y-4"
          >
            <h1 className="text-sm font-semibold text-zinc-200">Admin key required</h1>
            <input
              type="password"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              placeholder="Enter admin key"
              autoComplete="off"
              className="w-full bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-700 font-mono focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/20 transition-colors"
            />
            {error && <p className="text-xs text-red-400">{error}</p>}
            <button
              type="submit"
              disabled={loading || !keyInput.trim()}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-semibold text-sm rounded-xl px-4 py-3 transition-colors"
            >
              {loading ? "Checking…" : "Access Dashboard"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#08090e] text-white">
      {/* Header */}
      <header className="flex items-center justify-between px-8 py-4 border-b border-white/[0.07]">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-indigo-500 to-purple-600" />
          <span className="text-sm font-bold text-zinc-100">ATTICUS</span>
          <span className="text-xs text-zinc-500 bg-white/[0.05] border border-white/[0.08] rounded px-2 py-0.5">Cost Dashboard</span>
        </div>
        <div className="flex items-center gap-4">
          {loading && <span className="text-xs text-zinc-600">Refreshing…</span>}
          <button
            onClick={() => loadData(adminKey)}
            disabled={loading}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            ↺ Refresh
          </button>
          <button
            onClick={() => { sessionStorage.removeItem(ADMIN_KEY_STORAGE); setAuthed(false); setAdminKey(""); setData(null); }}
            className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            Sign out
          </button>
        </div>
      </header>

      <main className="px-8 py-8 max-w-5xl mx-auto space-y-8">
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 text-sm text-red-400">{error}</div>
        )}

        {data && (
          <>
            {/* KPIs */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl p-6">
                <div className="text-[0.6rem] font-bold text-zinc-600 tracking-[.1em] mb-2">TOTAL SPEND</div>
                <div className="text-4xl font-bold text-white tabular-nums">{fmt(data.total_cost_usd)}</div>
              </div>
              <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl p-6">
                <div className="text-[0.6rem] font-bold text-zinc-600 tracking-[.1em] mb-2">TOTAL QUERIES</div>
                <div className="text-4xl font-bold text-white tabular-nums">{data.total_queries}</div>
              </div>
            </div>

            {/* Per-user */}
            <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl overflow-hidden">
              <div className="px-6 py-4 border-b border-white/[0.07]">
                <h2 className="text-sm font-semibold text-zinc-300">Per User</h2>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/[0.05]">
                    <th className="text-left px-6 py-3 text-zinc-600 font-normal">User</th>
                    <th className="text-right px-6 py-3 text-zinc-600 font-normal">Queries</th>
                    <th className="text-right px-6 py-3 text-zinc-600 font-normal">Cost</th>
                    <th className="text-right px-6 py-3 text-zinc-600 font-normal">Last Active</th>
                  </tr>
                </thead>
                <tbody>
                  {data.by_user.map((u, i) => (
                    <tr key={u.user_slug ? `user-${u.user_slug}` : `user-row-${i}`} className="border-b border-white/[0.04] hover:bg-white/[0.02] transition-colors">
                      <td className="px-6 py-3 font-mono text-indigo-300">{u.user_slug || "—"}</td>
                      <td className="px-6 py-3 text-right text-zinc-400 tabular-nums">{u.query_count}</td>
                      <td className="px-6 py-3 text-right text-zinc-300 tabular-nums">{fmt(u.cost_usd)}</td>
                      <td className="px-6 py-3 text-right text-zinc-600">{fmtDate(u.last_active)}</td>
                    </tr>
                  ))}
                  {data.by_user.length === 0 && (
                    <tr><td colSpan={4} className="px-6 py-6 text-center text-zinc-700">No activity yet</td></tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Per-model */}
            <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl overflow-hidden">
              <div className="px-6 py-4 border-b border-white/[0.07]">
                <h2 className="text-sm font-semibold text-zinc-300">Per Model</h2>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/[0.05]">
                    <th className="text-left px-6 py-3 text-zinc-600 font-normal">Model</th>
                    <th className="text-right px-6 py-3 text-zinc-600 font-normal">Input Tokens</th>
                    <th className="text-right px-6 py-3 text-zinc-600 font-normal">Output Tokens</th>
                    <th className="text-right px-6 py-3 text-zinc-600 font-normal">Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {data.by_model.map((m, i) => (
                    <tr key={m.model ? `model-${m.model}` : `model-row-${i}`} className="border-b border-white/[0.04] hover:bg-white/[0.02] transition-colors">
                      <td className="px-6 py-3 font-mono text-zinc-400">{m.model || "—"}</td>
                      <td className="px-6 py-3 text-right text-zinc-500 tabular-nums">{m.input_tokens.toLocaleString()}</td>
                      <td className="px-6 py-3 text-right text-zinc-500 tabular-nums">{m.output_tokens.toLocaleString()}</td>
                      <td className="px-6 py-3 text-right text-zinc-300 tabular-nums">{fmt(m.cost_usd)}</td>
                    </tr>
                  ))}
                  {data.by_model.length === 0 && (
                    <tr><td colSpan={4} className="px-6 py-6 text-center text-zinc-700">No model usage yet</td></tr>
                  )}
                </tbody>
              </table>
            </div>

            {/* Recent queries */}
            <div className="bg-white/[0.03] border border-white/[0.08] rounded-2xl overflow-hidden">
              <div className="px-6 py-4 border-b border-white/[0.07]">
                <h2 className="text-sm font-semibold text-zinc-300">Recent Queries (last 20)</h2>
              </div>
              <div className="divide-y divide-white/[0.04]">
                {data.recent_queries.map((q, i) => (
                  <div key={i} className="px-6 py-3 hover:bg-white/[0.02] transition-colors">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[0.65rem] font-mono text-indigo-400">{q.user_slug}</span>
                      <div className="flex items-center gap-4">
                        <span className="text-[0.6rem] text-zinc-600">{fmtDate(q.timestamp)}</span>
                        <span className="text-[0.6rem] text-zinc-500 tabular-nums">{fmt(q.cost_usd)}</span>
                      </div>
                    </div>
                    <p className="text-xs text-zinc-500 leading-snug line-clamp-2">{q.question}</p>
                  </div>
                ))}
                {data.recent_queries.length === 0 && (
                  <div className="px-6 py-6 text-center text-zinc-700 text-xs">No queries yet</div>
                )}
              </div>
            </div>

            <p className="text-[0.6rem] text-zinc-700 text-center pb-4">
              Auto-refreshes every 30 seconds · {new Date().toLocaleTimeString()}
            </p>
          </>
        )}
      </main>
    </div>
  );
}
