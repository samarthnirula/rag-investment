"use client";

import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import {
  listOrgs,
  createOrg,
  renameOrg,
  listOrgMembers,
  addOrgMember,
  removeOrgMember,
  Org,
  OrgMember,
} from "@/lib/api";

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

function RoleBadge({ role }: { role: string }) {
  const styles: Record<string, string> = {
    owner:  "bg-amber-500/15 text-amber-300 border-amber-500/30",
    admin:  "bg-purple-500/15 text-purple-300 border-purple-500/30",
    member: "bg-zinc-500/10 text-zinc-400 border-zinc-500/20",
  };
  return (
    <span className={`text-[0.6rem] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full border ${styles[role] ?? styles.member}`}>
      {role}
    </span>
  );
}

export default function OrgPage() {
  const { user, idToken, loading } = useAuth();
  const router = useRouter();

  const [orgs, setOrgs] = useState<Org[]>([]);
  const [orgsLoading, setOrgsLoading] = useState(false);
  const [selectedOrgId, setSelectedOrgId] = useState<string | null>(null);
  const [members, setMembers] = useState<OrgMember[]>([]);
  const [membersLoading, setMembersLoading] = useState(false);

  const [showCreate, setShowCreate] = useState(false);
  const [newOrgName, setNewOrgName] = useState("");
  const [creating, setCreating] = useState(false);

  const [inviteUid, setInviteUid] = useState("");
  const [inviteRole, setInviteRole] = useState<"member" | "admin">("member");
  const [inviting, setInviting] = useState(false);

  const [renaming, setRenaming] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [showRename, setShowRename] = useState(false);

  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/");
  }, [user, loading, router]);

  const loadOrgs = useCallback(async () => {
    if (!idToken) return;
    setOrgsLoading(true);
    try {
      const data = await listOrgs(idToken);
      setOrgs(data);
      if (data.length > 0 && !selectedOrgId) {
        setSelectedOrgId(data[0].org_id);
      }
    } catch {
      setError("Failed to load organizations.");
    } finally {
      setOrgsLoading(false);
    }
  }, [idToken, selectedOrgId]);

  const loadMembers = useCallback(async (orgId: string) => {
    if (!idToken) return;
    setMembersLoading(true);
    try {
      const data = await listOrgMembers(idToken, orgId);
      setMembers(data);
    } catch {
      setMembers([]);
    } finally {
      setMembersLoading(false);
    }
  }, [idToken]);

  useEffect(() => {
    if (idToken) loadOrgs();
  }, [idToken, loadOrgs]);

  useEffect(() => {
    if (selectedOrgId) loadMembers(selectedOrgId);
  }, [selectedOrgId, loadMembers]);

  const selectedOrg = orgs.find((o) => o.org_id === selectedOrgId);
  const isOwner = selectedOrg?.owner_id === user?.uid;
  const isAdmin = isOwner || members.some(
    (m) => m.user_id === user?.uid && ["owner", "admin"].includes(m.role)
  );

  async function handleCreateOrg(e: React.FormEvent) {
    e.preventDefault();
    if (!idToken || !newOrgName.trim()) return;
    setCreating(true);
    setError(null);
    try {
      await createOrg(idToken, newOrgName.trim());
      setNewOrgName("");
      setShowCreate(false);
      await loadOrgs();
      setSuccess("Organization created.");
    } catch {
      setError("Failed to create organization.");
    } finally {
      setCreating(false);
    }
  }

  async function handleRenameOrg(e: React.FormEvent) {
    e.preventDefault();
    if (!idToken || !selectedOrgId || !renameValue.trim()) return;
    setRenaming(true);
    try {
      await renameOrg(idToken, selectedOrgId, renameValue.trim());
      setShowRename(false);
      await loadOrgs();
      setSuccess("Organization renamed.");
    } catch {
      setError("Failed to rename organization.");
    } finally {
      setRenaming(false);
    }
  }

  async function handleInvite(e: React.FormEvent) {
    e.preventDefault();
    if (!idToken || !selectedOrgId || !inviteUid.trim()) return;
    if (inviteUid.trim() === user?.uid) {
      setError("You can't invite yourself.");
      return;
    }
    setInviting(true);
    setError(null);
    try {
      await addOrgMember(idToken, selectedOrgId, inviteUid.trim(), inviteRole);
      setInviteUid("");
      await loadMembers(selectedOrgId);
      setSuccess(`User added as ${inviteRole}.`);
    } catch {
      setError("Failed to add member. Make sure the user ID is valid.");
    } finally {
      setInviting(false);
    }
  }

  async function handleRemoveMember(memberId: string) {
    if (!idToken || !selectedOrgId) return;
    if (!confirm("Remove this member?")) return;
    try {
      await removeOrgMember(idToken, selectedOrgId, memberId);
      await loadMembers(selectedOrgId);
    } catch {
      setError("Failed to remove member.");
    }
  }

  function formatJoinDate(iso: string | null) {
    if (!iso) return "";
    try {
      return new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    } catch {
      return iso;
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
    <div className="px-8 py-8 overflow-y-auto">
      <div className="max-w-3xl">
        <div className="mb-6">
          <h1 className="text-xl font-bold text-white">Team & Organization</h1>
          <p className="text-sm text-zinc-500 mt-0.5">
            Create a team workspace to share documents and cases with colleagues.
          </p>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-2.5 text-xs text-red-300 mb-4">
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-red-400">×</button>
          </div>
        )}
        {success && (
          <div className="bg-emerald-500/10 border border-emerald-500/25 rounded-lg px-4 py-2.5 text-xs text-emerald-300 mb-4">
            {success}
            <button onClick={() => setSuccess(null)} className="ml-2 text-emerald-400">×</button>
          </div>
        )}

        {/* Create org */}
        {!showCreate ? (
          <button
            onClick={() => setShowCreate(true)}
            className="mb-6 text-xs text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/10 rounded-lg px-4 py-2 transition-colors"
          >
            + Create new organization
          </button>
        ) : (
          <form onSubmit={handleCreateOrg} className="bg-white/[0.02] border border-white/[0.07] rounded-xl p-4 mb-6 space-y-3">
            <p className="text-sm font-semibold text-white">New Organization</p>
            <input
              autoFocus
              value={newOrgName}
              onChange={(e) => setNewOrgName(e.target.value)}
              placeholder="Acme Legal Group"
              className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50"
            />
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={creating || !newOrgName.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-4 py-2 transition-colors"
              >
                {creating ? "Creating…" : "Create"}
              </button>
              <button
                type="button"
                onClick={() => setShowCreate(false)}
                className="text-xs text-zinc-600 hover:text-zinc-400 border border-white/[0.07] rounded-lg px-4 py-2"
              >
                Cancel
              </button>
            </div>
          </form>
        )}

        {/* Org selector */}
        {orgsLoading && orgs.length === 0 && (
          <div className="flex py-8 justify-center">
            <Spinner />
          </div>
        )}

        {!orgsLoading && orgs.length === 0 && (
          <div className="text-center py-12 text-sm text-zinc-600">
            You are not a member of any organization yet.
          </div>
        )}

        {orgs.length > 0 && (
          <>
            {/* Org tabs */}
            {orgs.length > 1 && (
              <div className="flex gap-1 mb-5 flex-wrap">
                {orgs.map((o) => (
                  <button
                    key={o.org_id}
                    onClick={() => setSelectedOrgId(o.org_id)}
                    className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-colors ${
                      selectedOrgId === o.org_id
                        ? "bg-indigo-500/15 text-indigo-300"
                        : "text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04]"
                    }`}
                  >
                    {o.org_name}
                  </button>
                ))}
              </div>
            )}

            {selectedOrg && (
              <div>
                {/* Org header */}
                <div className="flex items-center gap-3 mb-5">
                  <div className="w-9 h-9 rounded-lg bg-indigo-500/20 flex items-center justify-center text-base font-bold text-indigo-300">
                    {selectedOrg.org_name[0]?.toUpperCase()}
                  </div>
                  <div>
                    <h2 className="text-base font-bold text-white">{selectedOrg.org_name}</h2>
                    <p className="text-xs text-zinc-500">
                      {members.length} member{members.length !== 1 ? "s" : ""}
                    </p>
                  </div>
                  {isOwner && (
                    <button
                      onClick={() => { setShowRename(true); setRenameValue(selectedOrg.org_name); }}
                      className="ml-auto text-xs text-zinc-600 hover:text-zinc-400 border border-white/[0.07] rounded-lg px-3 py-1.5 transition-colors"
                    >
                      Rename
                    </button>
                  )}
                </div>

                {/* Rename form */}
                {showRename && (
                  <form onSubmit={handleRenameOrg} className="flex gap-2 mb-5">
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      className="flex-1 bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50"
                    />
                    <button
                      type="submit"
                      disabled={renaming || !renameValue.trim()}
                      className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-4 py-2 transition-colors"
                    >
                      {renaming ? "Saving…" : "Save"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowRename(false)}
                      className="text-xs text-zinc-600 hover:text-zinc-400 border border-white/[0.07] rounded-lg px-3 py-2"
                    >
                      Cancel
                    </button>
                  </form>
                )}

                {/* Members list */}
                <div className="mb-5">
                  <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3">MEMBERS</p>
                  {membersLoading && <div className="flex py-4"><Spinner /></div>}
                  {!membersLoading && members.length === 0 && (
                    <p className="text-xs text-zinc-600">No members yet.</p>
                  )}
                  {members.map((m) => (
                    <div
                      key={m.user_id}
                      className="flex items-center justify-between bg-white/[0.02] border border-white/[0.06] rounded-lg px-4 py-3 mb-1.5"
                    >
                      <div className="min-w-0">
                        <p className="text-xs text-zinc-300 font-mono truncate">
                          {m.user_id.length > 28 ? m.user_id.slice(0, 28) + "…" : m.user_id}
                        </p>
                        {m.joined_at && (
                          <p className="text-[0.65rem] text-zinc-600 mt-0.5">
                            Joined {formatJoinDate(m.joined_at)}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-3 shrink-0 ml-4">
                        <RoleBadge role={m.role} />
                        {isAdmin && m.user_id !== user?.uid && m.role !== "owner" && (
                          <button
                            onClick={() => handleRemoveMember(m.user_id)}
                            className="text-xs text-zinc-600 hover:text-red-400 transition-colors"
                          >
                            Remove
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Invite */}
                {isAdmin && (
                  <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-4">
                    <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3">
                      INVITE MEMBER
                    </p>
                    <form onSubmit={handleInvite} className="space-y-3">
                      <div>
                        <label className="block text-xs text-zinc-500 mb-1">
                          Firebase UID
                        </label>
                        <input
                          value={inviteUid}
                          onChange={(e) => setInviteUid(e.target.value)}
                          placeholder="User ID (Firebase UID)"
                          className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50 font-mono"
                        />
                        <p className="text-[0.6rem] text-zinc-700 mt-1">
                          The Firebase UID of the user to invite. They must have already signed up.
                        </p>
                      </div>
                      <div className="flex gap-3 items-center">
                        <select
                          value={inviteRole}
                          onChange={(e) => setInviteRole(e.target.value as "member" | "admin")}
                          className="text-xs bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-zinc-400 focus:outline-none"
                        >
                          <option value="member">Member</option>
                          <option value="admin">Admin</option>
                        </select>
                        <button
                          type="submit"
                          disabled={inviting || !inviteUid.trim()}
                          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-5 py-2 transition-colors"
                        >
                          {inviting ? "Inviting…" : "Invite"}
                        </button>
                      </div>
                    </form>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
