/**
 * Atticus API client.
 *
 * Every call to a protected endpoint automatically attaches a fresh Firebase
 * ID token. Firebase's SDK silently refreshes the token when it is within
 * 5 minutes of expiry (tokens live for 1 hour), so callers never need to
 * manage tokens or pass them explicitly.
 */

import { auth } from "@/lib/firebase";

const DEFAULT_BASE =
  process.env.NODE_ENV === "production"
    ? "https://atticus-backend.onrender.com"
    : "http://localhost:8000";
const CONFIGURED_BASE = process.env.NEXT_PUBLIC_API_URL?.trim() || "";
const BASE = (
  process.env.NODE_ENV === "production" && /^https?:\/\/(?:localhost|127\.0\.0\.1)(?::\d+)?$/i.test(CONFIGURED_BASE)
    ? DEFAULT_BASE
    : CONFIGURED_BASE || DEFAULT_BASE
).replace(/\/+$/, "");

/** Fetch a fresh (auto-refreshed) Firebase ID token for the current user. */
async function getToken(): Promise<string> {
  const user = auth?.currentUser;
  if (!user) return "";
  // false = use cached token if still valid; SDK refreshes automatically
  // when < 5 min remain. Pass true to force-refresh.
  return user.getIdToken(false);
}

async function request<T>(path: string, opts: RequestInit = {}, _retry = false): Promise<T> {
  const token = await getToken();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(opts.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers,
    credentials: "include",
  });

  // On 401, force-refresh the Firebase ID token and retry once.
  if (res.status === 401 && !_retry) {
    const user = auth?.currentUser;
    if (user) {
      try { await user.getIdToken(true); } catch { /* ignore */ }
    }
    return request<T>(path, opts, true);
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface SessionUser {
  uid: string;
  email: string;
  display_name: string;
}

export async function getSession(): Promise<SessionUser> {
  return request<SessionUser>("/api/auth/session", { method: "POST" });
}

export interface RegisterResult {
  uid: string;
  plan: string;
  is_new: boolean;
}

/** Idempotent — call on every sign-in to sync the user to Postgres and Zep. */
export async function registerUser(): Promise<RegisterResult> {
  return request<RegisterResult>("/api/auth/register", { method: "POST", body: "{}" });
}

// ── Chats ─────────────────────────────────────────────────────────────────────

export interface ChatSummary {
  chat_id: string;
  name: string;
  page: string;
  case_id?: string | null;
  chat_type?: string | null;
}

export async function listChats(): Promise<ChatSummary[]> {
  return request<ChatSummary[]>("/api/chats");
}

export async function createChat(
  payloadOrToken: { chat_id?: string; page?: string; name?: string; case_id?: string | null } | string = {},
  maybePayload?: { chat_id?: string; page?: string; name?: string; case_id?: string | null }
): Promise<{ chat_id: string }> {
  const payload = typeof payloadOrToken === "string" ? maybePayload ?? {} : payloadOrToken;
  return request<{ chat_id: string }>("/api/chats", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deleteChat(
  chatIdOrToken: string,
  maybeChatId?: string
): Promise<{ ok: boolean }> {
  const chatId = maybeChatId ?? chatIdOrToken;
  return request<{ ok: boolean }>(`/api/chats/${chatId}`, { method: "DELETE" });
}

export async function renameChat(
  chatIdOrToken: string,
  nameOrChatId: string,
  maybeName?: string
): Promise<{ ok: boolean }> {
  const chatId = maybeName ? nameOrChatId : chatIdOrToken;
  const name = maybeName ?? nameOrChatId;
  return request<{ ok: boolean }>(`/api/chats/${chatId}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
  });
}

// ── Messages ──────────────────────────────────────────────────────────────────

export interface Confidence {
  score: number;
  rating: string;
  rationale: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  images?: ImageAttachment[];
  imageNote?: string | null;
  sourceDetails?: SourceDetail[];
  query?: string;
  confidence?: Confidence;
}

export async function getMessages(chatIdOrToken: string, maybeChatId?: string): Promise<Message[]> {
  const chatId = maybeChatId ?? chatIdOrToken;
  return request<Message[]>(`/api/chats/${chatId}/messages`);
}

export async function saveMessage(
  chatIdOrToken: string,
  payloadOrChatId: { role: string; content: string } | string,
  maybePayload?: { role: string; content: string }
): Promise<{ ok: boolean }> {
  const chatId = typeof payloadOrChatId === "string" ? payloadOrChatId : chatIdOrToken;
  const payload = typeof payloadOrChatId === "string" ? maybePayload : payloadOrChatId;
  return request<{ ok: boolean }>(`/api/chats/${chatId}/messages`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ── Companies ──────────────────────────────────────────────────────────────────

export async function listCompanies(_token?: string): Promise<string[]> {
  return request<string[]>("/api/companies");
}

// ── Query ─────────────────────────────────────────────────────────────────────

export interface QueryResponse {
  answer: string;
  sources: string[];
  source_details?: SourceDetail[];
  images?: ImageAttachment[];
  image_note?: string | null;
}

export interface SourceDetail {
  index: number;
  label: string;
  chunk_id: string;
  document_id: string;
  file_name: string;
  page_number: number;
  section_header: string | null;
  chunk_text: string;
  excerpt?: string;
  source_type?: string;
  citation_label?: string;
  jurisdiction?: string | null;
}

export interface ImageAttachment {
  image_id: string;
  url: string;
  source: string;
  document_id: string;
  page_number: number;
  image_index: number;
  width: number | null;
  height: number | null;
  description: string | null;
}

export function apiAssetUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${BASE}${path}`;
}

export async function fetchImageObjectUrl(path: string): Promise<string> {
  const token = await getToken();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(apiAssetUrl(path), { headers, credentials: "include" });
  if (!res.ok) throw new Error("Image could not be loaded.");
  return URL.createObjectURL(await res.blob());
}

export async function runQuery(
  payloadOrToken: {
    query: string;
    chat_id?: string;
    page?: string;
    company_filter?: string;
    top_k?: number;
    case_id?: string;
  } | string,
  maybePayload?: {
    query: string;
    chat_id?: string;
    page?: string;
    company_filter?: string;
    top_k?: number;
    case_id?: string;
  }
): Promise<QueryResponse> {
  const payload = typeof payloadOrToken === "string" ? maybePayload : payloadOrToken;
  return request<QueryResponse>("/api/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runDemoQuery(payload: {
  query: string;
  top_k?: number;
}): Promise<QueryResponse> {
  return request<QueryResponse>("/api/demo/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface StreamResult {
  text: string;
  sources: string[];
  source_details?: SourceDetail[];
  images?: ImageAttachment[];
  image_note?: string | null;
  confidence?: Confidence;
}

/**
 * POST /api/query/stream — streams SSE tokens, resolves with final result.
 * Throws an Error whose message is safe to show directly to the user.
 */
export async function runQueryStream(
  payload: {
    query: string;
    chat_id?: string;
    page?: string;
    company_filter?: string;
    top_k?: number;
    case_id?: string;
  },
  onToken: (token: string) => void,
): Promise<StreamResult> {
  const token = await getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(`${BASE}/api/query/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      credentials: "include",
    });
  } catch {
    throw new Error("Network error — check your connection and try again.");
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("Your browser does not support response streaming.");

  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";
  let finalResult: StreamResult | null = null;

  try {
    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let data: Record<string, unknown>;
        try { data = JSON.parse(raw); } catch { continue; }

        if (typeof data.error === "string") throw new Error(data.error);

        if (typeof data.token === "string") {
          fullText += data.token;
          onToken(data.token);
        }

        if (data.done) {
          finalResult = {
            // Backend sends cleaned text (confidence block stripped) in data.text
            text: typeof data.text === "string" ? data.text : fullText,
            sources: (data.sources as string[]) ?? [],
            source_details: data.source_details as SourceDetail[] | undefined,
            images: data.images as ImageAttachment[] | undefined,
            image_note: data.image_note as string | null | undefined,
            confidence: data.confidence as Confidence | undefined,
          };
          break outer;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  return finalResult ?? { text: fullText, sources: [] };
}

// ── Usage ─────────────────────────────────────────────────────────────────────

export interface UsageStats {
  queries_today: number;
  queries_this_month: number;
  queries_limit: number;
}

export async function getUsage(_token?: string): Promise<UsageStats> {
  return request<UsageStats>("/api/usage");
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function healthCheck(): Promise<{ status: string }> {
  return request<{ status: string }>("/api/health");
}

// ── Cases ─────────────────────────────────────────────────────────────────────

export interface Case {
  case_id: string;
  case_name: string;
  description: string | null;
  document_count: number;
}

export interface CaseDocument {
  document_id: string;
  file_name: string;
  company: string | null;
}

export async function listCases(_token?: string): Promise<Case[]> {
  return request<Case[]>("/api/cases");
}

export async function createCase(
  payloadOrToken: { case_name: string; description?: string } | string,
  maybePayload?: { case_name: string; description?: string }
): Promise<{ case_id: string }> {
  const payload = typeof payloadOrToken === "string" ? maybePayload : payloadOrToken;
  return request<{ case_id: string }>("/api/cases", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deleteCase(caseIdOrToken: string, maybeCaseId?: string): Promise<{ ok: boolean }> {
  const caseId = maybeCaseId ?? caseIdOrToken;
  return request<{ ok: boolean }>(`/api/cases/${caseId}`, { method: "DELETE" });
}

export async function getCaseDocuments(caseIdOrToken: string, maybeCaseId?: string): Promise<CaseDocument[]> {
  const caseId = maybeCaseId ?? caseIdOrToken;
  return request<CaseDocument[]>(`/api/cases/${caseId}/documents`);
}

export async function addDocumentToCase(
  caseIdOrToken: string,
  documentIdOrCaseId: string,
  maybeDocumentId?: string
): Promise<{ ok: boolean }> {
  const caseId = maybeDocumentId ? documentIdOrCaseId : caseIdOrToken;
  const documentId = maybeDocumentId ?? documentIdOrCaseId;
  return request<{ ok: boolean }>(`/api/cases/${caseId}/documents`, {
    method: "POST",
    body: JSON.stringify({ document_id: documentId }),
  });
}

export async function removeDocumentFromCase(
  caseIdOrToken: string,
  docIdOrCaseId: string,
  maybeDocId?: string
): Promise<{ ok: boolean }> {
  const caseId = maybeDocId ? docIdOrCaseId : caseIdOrToken;
  const docId = maybeDocId ?? docIdOrCaseId;
  return request<{ ok: boolean }>(`/api/cases/${caseId}/documents/${docId}`, {
    method: "DELETE",
  });
}

// ── Documents ─────────────────────────────────────────────────────────────────

export interface Document {
  document_id: string;
  file_name: string;
  company: string | null;
  document_type: string | null;
  page_count: number;
  version_label: string | null;
  is_system: boolean;
}

export async function listDocuments(_token?: string): Promise<Document[]> {
  return request<Document[]>("/api/documents");
}

export async function deleteDocument(docIdOrToken: string, maybeDocId?: string): Promise<{ ok: boolean }> {
  const docId = maybeDocId ?? docIdOrToken;
  return request<{ ok: boolean }>(`/api/documents/${docId}`, { method: "DELETE" });
}

export async function uploadDocument(
  fileOrToken: File | string,
  maybeFile?: File
): Promise<{ document_id: string; file_name: string; page_count: number; chunks_inserted: number }> {
  const file = maybeFile ?? fileOrToken;
  if (typeof file === "string") throw new Error("Upload file is required");
  const token = await getToken();
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/api/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
    credentials: "include",
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

// ── Bulk upload ───────────────────────────────────────────────────────────────

export interface BulkUploadResult {
  case_id: string;
  job_ids: string[];
  total_files: number;
  skipped: string[];
}

export interface CaseJobStatus {
  total: number;
  completed: number;
  failed: number;
  in_progress: number;
  pending: number;
  errors: string[];
}

export async function bulkUploadCase(
  files: File[],
  meta: { case_name: string; matter_type?: string; jurisdiction?: string; client_name?: string; notes?: string }
): Promise<BulkUploadResult> {
  const token = await getToken();
  const form = new FormData();
  for (const file of files) {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
    form.append("files", file, relativePath || file.name);
  }
  form.append("case_name", meta.case_name);
  if (meta.matter_type) form.append("matter_type", meta.matter_type);
  if (meta.jurisdiction) form.append("jurisdiction", meta.jurisdiction);
  if (meta.client_name) form.append("client_name", meta.client_name);
  if (meta.notes) form.append("notes", meta.notes);

  const res = await fetch(`${BASE}/api/cases/bulk-upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
    credentials: "include",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Bulk upload failed");
  }
  return res.json();
}

export async function getCaseJobs(caseId: string): Promise<CaseJobStatus> {
  return request<CaseJobStatus>(`/api/cases/${caseId}/jobs`);
}

// ── Discussion ────────────────────────────────────────────────────────────────

export interface Post {
  post_id: string;
  type: string;
  author: string;
  content: string;
  time: string;
}

export async function listPosts(_token?: string): Promise<Post[]> {
  return request<Post[]>("/api/discussion");
}

export async function createPost(
  payloadOrToken: { post_type: string; content: string } | string,
  maybePayload?: { post_type: string; content: string }
): Promise<{ ok: boolean }> {
  const payload = typeof payloadOrToken === "string" ? maybePayload : payloadOrToken;
  return request<{ ok: boolean }>("/api/discussion", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deletePost(postIdOrToken: string, maybePostId?: string): Promise<{ ok: boolean }> {
  const postId = maybePostId ?? postIdOrToken;
  return request<{ ok: boolean }>(`/api/discussion/${postId}`, {
    method: "DELETE",
  });
}

// ── Organizations ─────────────────────────────────────────────────────────────

export interface Org {
  org_id: string;
  org_name: string;
  owner_id: string;
}

export interface OrgMember {
  user_id: string;
  role: string;
  joined_at: string | null;
}

export async function listOrgs(_token?: string): Promise<Org[]> {
  return request<Org[]>("/api/orgs");
}

export async function createOrg(orgNameOrToken: string, maybeOrgName?: string): Promise<{ ok: boolean }> {
  const orgName = maybeOrgName ?? orgNameOrToken;
  return request<{ ok: boolean }>("/api/orgs", {
    method: "POST",
    body: JSON.stringify({ org_name: orgName }),
  });
}

export async function renameOrg(
  orgIdOrToken: string,
  orgNameOrOrgId: string,
  maybeOrgName?: string
): Promise<{ ok: boolean }> {
  const orgId = maybeOrgName ? orgNameOrOrgId : orgIdOrToken;
  const orgName = maybeOrgName ?? orgNameOrOrgId;
  return request<{ ok: boolean }>(`/api/orgs/${orgId}`, {
    method: "PATCH",
    body: JSON.stringify({ org_name: orgName }),
  });
}

export async function listOrgMembers(orgIdOrToken: string, maybeOrgId?: string): Promise<OrgMember[]> {
  const orgId = maybeOrgId ?? orgIdOrToken;
  return request<OrgMember[]>(`/api/orgs/${orgId}/members`);
}

export async function addOrgMember(
  orgIdOrToken: string,
  userIdOrOrgId: string,
  roleOrUserId: string,
  maybeRole?: string
): Promise<{ ok: boolean }> {
  const orgId = maybeRole ? userIdOrOrgId : orgIdOrToken;
  const userId = maybeRole ? roleOrUserId : userIdOrOrgId;
  const role = maybeRole ?? roleOrUserId;
  return request<{ ok: boolean }>(`/api/orgs/${orgId}/members`, {
    method: "POST",
    body: JSON.stringify({ user_id: userId, role }),
  });
}

export async function removeOrgMember(
  orgIdOrToken: string,
  memberIdOrOrgId: string,
  maybeMemberId?: string
): Promise<{ ok: boolean }> {
  const orgId = maybeMemberId ? memberIdOrOrgId : orgIdOrToken;
  const memberId = maybeMemberId ?? memberIdOrOrgId;
  return request<{ ok: boolean }>(`/api/orgs/${orgId}/members/${memberId}`, {
    method: "DELETE",
  });
}

// ── Analytics ─────────────────────────────────────────────────────────────────

export interface AnalyticsData {
  stats: {
    total_queries: number;
    queries_this_month: number;
    queries_today: number;
    chunks_retrieved: number;
    estimated_cost_usd: number;
  };
  daily: Array<{ date: string; Queries: number }>;
  by_page: Array<{ Page: string; Queries: number }>;
  recent: Array<{ Time: string; Chat: string; Query: string; Sources: number; "Resp len": number }>;
  uploads_this_month: number;
  upload_cost_this_month: number;
}

export async function getAnalytics(daysOrToken: number | string = 30, maybeDays?: number): Promise<AnalyticsData> {
  const days = maybeDays ?? (typeof daysOrToken === "number" ? daysOrToken : 30);
  return request<AnalyticsData>(`/api/analytics?days=${days}`);
}

// ── Profile delete ─────────────────────────────────────────────────────────────

export async function deleteProfile(_token?: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/api/profile", { method: "DELETE" });
}

// ── Subscription ──────────────────────────────────────────────────────────────

export interface SubscriptionStatus {
  plan: string;
  trial_expires_at: string | null;
  days_remaining: number;
  hours_remaining: number;
  is_trial_expired: boolean;
  subscription_active: boolean;
  case_count: number;
  trial_case_limit: number;
}

export async function getSubscriptionStatus(): Promise<SubscriptionStatus> {
  return request<SubscriptionStatus>("/api/subscription/status");
}

// ── Case tabs (chats / overview / timeline) ───────────────────────────────────

export interface CaseChat {
  chat_id: string;
  name: string;
  chat_type: "chat" | "timeline" | "overview";
}

export async function getCaseChats(caseId: string): Promise<CaseChat[]> {
  return request<CaseChat[]>(`/api/cases/${caseId}/chats`);
}

export interface CaseOverview {
  pending: boolean;
  summary?: string;
  parties?: Array<{ role: string; name: string }>;
  key_issues?: string[];
  jurisdiction?: string | null;
  matter_type?: string | null;
  generated_at?: string | null;
}

export async function getCaseOverview(caseId: string): Promise<CaseOverview> {
  return request<CaseOverview>(`/api/cases/${caseId}/overview`);
}

export interface CaseTimelineEvent {
  date: string;
  title: string;
  description: string;
  source_doc?: string;
  page?: number;
}

export interface CaseTimeline {
  pending: boolean;
  events?: CaseTimelineEvent[];
  generated_at?: string | null;
}

export async function getCaseTimeline(caseId: string): Promise<CaseTimeline> {
  return request<CaseTimeline>(`/api/cases/${caseId}/timeline`);
}
