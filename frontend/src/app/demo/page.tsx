"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { demoAuth, setDemoToken, getDemoToken } from "@/lib/demo-api";

export default function DemoEntryPage() {
  const router = useRouter();
  const [code, setCode]       = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [shake, setShake]     = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (getDemoToken()) router.replace("/demo/chat");
  }, [router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = code.trim().toUpperCase();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");
    try {
      const result = await demoAuth(trimmed);
      setDemoToken(result.token);
      router.push("/demo/chat");
    } catch {
      setError("Invalid access code. Please try again.");
      setShake(true);
      setTimeout(() => setShake(false), 600);
      setCode("");
      inputRef.current?.focus();
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#08090e] flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="flex items-center justify-center gap-3 mb-10">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/20" />
          <span className="text-2xl font-bold text-white tracking-tight">ATTICUS</span>
        </div>

        <div
          className={`bg-white/[0.03] border border-white/[0.09] rounded-2xl px-8 py-8 ${shake ? "animate-shake" : ""}`}
          style={{ boxShadow: "0 24px 48px rgba(0,0,0,0.4)" }}
        >
          <h1 className="text-base font-semibold text-zinc-200 mb-1">Enter your access code</h1>
          <p className="text-xs text-zinc-600 mb-6">Epstein Case Demo — provided by Atticus</p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <input
              ref={inputRef}
              type="text"
              value={code}
              onChange={(e) => setCode(e.target.value.toUpperCase())}
              placeholder="XXXXXX"
              maxLength={12}
              autoComplete="off"
              spellCheck={false}
              disabled={loading}
              className="w-full bg-white/[0.05] border border-white/[0.1] rounded-xl px-4 py-3.5 text-base text-white placeholder-zinc-700 font-mono tracking-[0.25em] text-center focus:outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/20 disabled:opacity-50 transition-colors"
            />

            {error && (
              <p className="text-xs text-red-400 text-center">{error}</p>
            )}

            <button
              type="submit"
              disabled={loading || !code.trim()}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-semibold text-sm rounded-xl px-4 py-3.5 transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  Verifying…
                </span>
              ) : "Enter"}
            </button>
          </form>
        </div>

        <p className="text-[0.65rem] text-zinc-700 text-center mt-6">
          AI responses are for research purposes only — not legal advice.
        </p>
      </div>

      <style>{`
        @keyframes shake {
          0%,100% { transform: translateX(0); }
          20%,60%  { transform: translateX(-8px); }
          40%,80%  { transform: translateX(8px); }
        }
        .animate-shake { animation: shake 0.5s ease-in-out; }
      `}</style>
    </div>
  );
}
