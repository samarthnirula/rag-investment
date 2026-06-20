"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { getUsage, deleteProfile } from "@/lib/api";
import { signOut } from "@/lib/firebase";

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

const DELETE_PHRASE = "DELETE MY ACCOUNT";

export default function ProfilePage() {
  const { user, idToken, plan, loading } = useAuth();
  const router = useRouter();
  const [usage, setUsage] = useState<{
    queries_today: number;
    queries_this_month: number;
    queries_limit: number;
  } | null>(null);

  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [showDanger, setShowDanger] = useState(false);

  useEffect(() => {
    if (!loading && !user) router.push("/");
  }, [user, loading, router]);

  useEffect(() => {
    if (!idToken) return;
    getUsage(idToken).then(setUsage).catch(() => {});
  }, [idToken]);

  const deleteEnabled = deleteConfirm === DELETE_PHRASE;

  async function handleDeleteData() {
    if (!idToken || !user || !deleteEnabled) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteProfile(idToken);
      await signOut();
      router.push("/");
    } catch {
      setDeleteError("Deletion failed. Please try again.");
      setDeleting(false);
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

  const limit = usage?.queries_limit ?? 500;
  const today = usage?.queries_today ?? 0;
  const month = usage?.queries_this_month ?? 0;
  const todayPct = Math.min(100, (today / limit) * 100);
  const monthPct = Math.min(100, (month / (limit * 30)) * 100);

  return (
    <div className="px-8 py-8 max-w-2xl overflow-y-auto">
      <h1 className="text-xl font-bold text-white mb-6">Profile</h1>

      {/* User card */}
      <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-6 mb-5 flex items-center gap-4">
        <div className="w-12 h-12 rounded-full bg-indigo-500/30 flex items-center justify-center text-lg font-bold text-indigo-300 shrink-0">
          {user.email?.[0]?.toUpperCase() ?? "?"}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-white truncate">
            {user.displayName || "Anonymous"}
          </div>
          <div className="text-xs text-zinc-500 truncate">{user.email}</div>
          <div className="text-[0.65rem] text-zinc-700 mt-1 font-mono truncate">
            UID: {user.uid}
          </div>
        </div>
      </div>

      {/* Usage stats */}
      <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-6 mb-5">
        <div className="text-sm font-semibold text-white mb-4">Usage</div>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs text-zinc-500 mb-1.5">
              <span>Today</span>
              <span>{today} / {limit}</span>
            </div>
            <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500 rounded-full transition-all"
                style={{ width: `${todayPct}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs text-zinc-500 mb-1.5">
              <span>This month</span>
              <span>{month} queries</span>
            </div>
            <div className="h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full transition-all"
                style={{ width: `${monthPct}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Plan */}
      <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-6 mb-5">
        <div className="text-sm font-semibold text-white mb-1">Plan</div>
        <div className="text-xs text-zinc-500 mb-3">Subscription management and billing</div>
        <div className="bg-indigo-500/10 border border-indigo-500/25 rounded-lg px-4 py-3 flex items-center justify-between">
          <span className="text-xs text-indigo-300 font-medium uppercase tracking-wide">{plan}</span>
          <span className="text-xs text-zinc-600">Manage billing →</span>
        </div>
      </div>

      {/* Danger zone — GDPR deletion */}
      <div className="border border-red-500/20 rounded-xl p-5">
        <button
          onClick={() => setShowDanger((v) => !v)}
          className="flex items-center gap-2 text-xs text-red-400 font-semibold w-full text-left"
        >
          <span>⚠</span> Danger zone — delete all my data
          <span className="ml-auto text-red-500/50">{showDanger ? "▲" : "▼"}</span>
        </button>

        {showDanger && (
          <div className="mt-4 space-y-3">
            <p className="text-xs text-zinc-500">
              This will permanently delete all your query logs, saved chats, cases, and uploaded
              documents. This action cannot be undone.
            </p>
            <div>
              <label className="block text-xs text-zinc-500 mb-1.5">
                Type <span className="text-red-400 font-mono font-semibold">{DELETE_PHRASE}</span> to confirm:
              </label>
              <input
                value={deleteConfirm}
                onChange={(e) => setDeleteConfirm(e.target.value)}
                placeholder={DELETE_PHRASE}
                className={`w-full bg-white/[0.05] border rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-700 focus:outline-none transition-colors ${
                  deleteConfirm.length > 0 && !deleteEnabled
                    ? "border-red-500/50 focus:border-red-500/70"
                    : deleteEnabled
                    ? "border-emerald-500/50 focus:border-emerald-500/70"
                    : "border-red-500/30 focus:border-red-500/50"
                }`}
              />
              {deleteConfirm.length > 0 && !deleteEnabled && (
                <p className="text-[0.6rem] text-red-400 mt-1">
                  Type the exact phrase to enable the button.
                </p>
              )}
            </div>
            {deleteError && (
              <p className="text-xs text-red-400">{deleteError}</p>
            )}
            <button
              onClick={handleDeleteData}
              disabled={deleting || !deleteEnabled}
              className="bg-red-600 hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-lg px-5 py-2.5 transition-colors"
            >
              {deleting ? "Deleting…" : "Delete all my data"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
