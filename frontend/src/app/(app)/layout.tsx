"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { signOut } from "@/lib/firebase";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { listCases, type Case } from "@/lib/api";

const navItems = [
  { href: "/chat", label: "Chat", icon: "💬" },
  { href: "/epstein", label: "Epstein", icon: "⚖" },
  { href: "/cases", label: "Cases", icon: "📁" },
  { href: "/org", label: "Team", icon: "🏢" },
  { href: "/data", label: "Data", icon: "📊" },
  { href: "/profile", label: "Profile", icon: "👤" },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { user, loading, isTrialExpired, daysRemaining, hoursRemaining, subscriptionStatus } = useAuth();
  const router = useRouter();
  const [signingOut, setSigningOut] = useState(false);
  const [cases, setCases] = useState<Case[]>([]);
  const [workspace, setWorkspace] = useState("epstein");

  // Single auth guard for every app route — no need to repeat in each page.
  useEffect(() => {
    if (!loading && !user) router.replace("/");
  }, [user, loading, router]);

  useEffect(() => {
    if (!user) return;
    const stored = localStorage.getItem("atticus_active_workspace");
    if (stored) setWorkspace(stored);
    const refreshCases = () => listCases().then(setCases).catch(() => setCases([]));
    const syncWorkspace = (event: Event) => {
      const custom = event as CustomEvent<string>;
      if (custom.detail) setWorkspace(custom.detail);
    };
    refreshCases();
    window.addEventListener("atticus-cases-changed", refreshCases);
    window.addEventListener("atticus-workspace-sync", syncWorkspace);
    window.addEventListener("focus", refreshCases);
    return () => {
      window.removeEventListener("atticus-cases-changed", refreshCases);
      window.removeEventListener("atticus-workspace-sync", syncWorkspace);
      window.removeEventListener("focus", refreshCases);
    };
  }, [user]);

  // Blank screen while Firebase resolves auth state (avoids flash of content)
  if (loading || !user) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#08090e]">
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <div key={i} className="w-2 h-2 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
          ))}
        </div>
      </div>
    );
  }

  async function handleSignOut() {
    setSigningOut(true);
    await signOut();
    router.push("/");
  }

  function handleWorkspaceChange(value: string) {
    setWorkspace(value);
    localStorage.setItem("atticus_active_workspace", value);
    window.dispatchEvent(new CustomEvent("atticus-workspace-change", { detail: value }));
    if (!pathname.startsWith("/chat")) {
      router.push("/chat");
    }
  }

  return (
    <div className="flex h-screen bg-[#08090e] text-white">
      {/* Sidebar */}
      <aside className="w-56 flex flex-col border-r border-white/[0.07] shrink-0">
        {/* Logo */}
        <div className="flex items-center gap-2 px-5 py-4 border-b border-white/[0.06]">
          <div className="w-5 h-5 rounded bg-gradient-to-br from-indigo-500 to-purple-500" />
          <span className="text-sm font-bold text-zinc-100 tracking-tight">Atticus</span>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          <div className="mb-4 rounded-xl border border-white/[0.07] bg-white/[0.03] p-3">
            <label className="mb-1.5 block text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-zinc-600">
              Active Case
            </label>
            <select
              value={workspace}
              onChange={(e) => handleWorkspaceChange(e.target.value)}
              className="w-full rounded-lg border border-white/[0.08] bg-[#0d0e14] px-2 py-2 text-xs text-zinc-300 outline-none transition-colors focus:border-indigo-500/50"
            >
              <option value="epstein">Epstein&apos;s case</option>
              {cases.map((c) => (
                <option key={c.case_id} value={`case:${c.case_id}:${c.case_name}`}>
                  {c.case_name}
                </option>
              ))}
            </select>
            <p className="mt-1.5 text-[0.65rem] leading-relaxed text-zinc-700">
              Chat searches the selected corpus.
            </p>
          </div>

          {navItems.map(({ href, label, icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                  active
                    ? "bg-indigo-500/15 text-indigo-300 font-medium"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04]"
                }`}
              >
                <span>{icon}</span>
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Trial countdown widget */}
        {subscriptionStatus && !subscriptionStatus.subscription_active && (
          <div className="px-3 pb-3">
            {isTrialExpired ? (
              <div className="rounded-lg px-3 py-2 bg-red-500/15 border border-red-500/25 text-xs text-red-400 font-medium">
                Trial expired — upgrade to continue
              </div>
            ) : daysRemaining <= 1 ? (
              <div className="rounded-lg px-3 py-2 bg-orange-500/15 border border-orange-500/25 text-xs text-orange-400 font-medium">
                {daysRemaining === 0
                  ? `${hoursRemaining}h left in trial`
                  : `${daysRemaining}d ${hoursRemaining}h left in trial`}
              </div>
            ) : (
              <div className="rounded-lg px-3 py-2 bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-400">
                {daysRemaining}d left in trial
              </div>
            )}
          </div>
        )}

        {/* User footer */}
        <div className="px-4 py-4 border-t border-white/[0.06]">
          {user ? (
            <div className="flex items-center gap-2.5 mb-3">
              <div className="w-7 h-7 rounded-full bg-indigo-500/30 flex items-center justify-center text-xs font-semibold text-indigo-300">
                {user.email?.[0]?.toUpperCase() ?? "?"}
              </div>
              <div className="min-w-0">
                <div className="text-xs text-zinc-400 truncate">{user.displayName || user.email}</div>
              </div>
            </div>
          ) : null}
          <button
            onClick={handleSignOut}
            disabled={signingOut}
            className="w-full text-xs text-zinc-600 hover:text-zinc-400 border border-white/[0.07] rounded-lg px-3 py-1.5 transition-colors disabled:opacity-50"
          >
            {signingOut ? "Signing out…" : "Sign out"}
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
