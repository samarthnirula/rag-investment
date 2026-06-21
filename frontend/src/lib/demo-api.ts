/**
 * Demo portal API client.
 * Uses a demo JWT cached in browser storage — no Firebase.
 */

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
const TOKEN_KEY = "demo_token";
const REQUEST_TIMEOUT_MS = 20000;

async function withTimeout<T>(run: (signal: AbortSignal) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await run(controller.signal);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request timed out while waiting for the Atticus backend.");
    }
    throw err;
  } finally {
    window.clearTimeout(timeout);
  }
}

export function getDemoToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
}

export function setDemoToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearDemoToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(TOKEN_KEY);
}

export function demoAssetUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${BASE}${path}`;
}

async function demoRequest<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const token = getDemoToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(opts.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await withTimeout((signal) => fetch(`${BASE}${path}`, { ...opts, headers, signal }));

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export interface DemoContactInfo {
  first_name?: string;
  last_name?: string;
  email?: string;
  phone?: string;
}

export async function demoAuth(
  accessCode: string,
  contact?: DemoContactInfo
): Promise<{ user_slug: string; token: string; ok: boolean }> {
  return demoRequest("/api/demo/auth", {
    method: "POST",
    body: JSON.stringify({ access_code: accessCode, ...contact }),
  });
}

export async function demoMe(): Promise<{ user_slug: string; query_count: number; last_active: string | null }> {
  return demoRequest("/api/demo/me");
}

// ── Demo Cases ───────────────────────────────────────────────────────────────

export interface DemoCase {
  case_id: string;
  case_name: string;
  description: string | null;
  document_count: number;
}

export interface DemoUploadResult {
  case_id: string;
  case_name: string;
  documents: Array<{
    document_id: string;
    file_name: string;
    page_count: number;
    chunks_inserted: number;
  }>;
  skipped: string[];
}

export async function demoCases(): Promise<DemoCase[]> {
  return demoRequest("/api/demo/cases");
}

export async function demoUploadCase(files: File[], caseName: string): Promise<DemoUploadResult> {
  const token = getDemoToken();
  const form = new FormData();
  for (const file of files) {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
    form.append("files", file, relativePath || file.name);
  }
  form.append("case_name", caseName);

  const res = await withTimeout((signal) =>
    fetch(`${BASE}/api/demo/cases/upload`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: form,
      signal,
    })
  );

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Demo upload failed");
  }
  return res.json() as Promise<DemoUploadResult>;
}

// ── Query ─────────────────────────────────────────────────────────────────────

export interface DemoSourceDetail {
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

export interface DemoQueryResponse {
  answer: string;
  sources: string[];
  source_details: DemoSourceDetail[];
  cost_usd: number;
  confidence?: {
    score: number;
    rating: string;
    rationale: string;
  };
}

export async function demoQuery(
  question: string,
  chatHistory: Array<{ role: string; content: string }> = [],
  caseId?: string | null
): Promise<DemoQueryResponse> {
  return demoRequest("/api/demo/query", {
    method: "POST",
    body: JSON.stringify({ question, chat_history: chatHistory, case_id: caseId || undefined }),
  });
}

// ── Timeline & Overview ───────────────────────────────────────────────────────

export interface DemoTimelineEvent {
  date: string;
  title: string;
  description: string;
  source_doc?: string;
  page?: number;
  image?: {
    image_id: string;
    url: string;
    source: string;
    document_id: string;
    page_number: number;
    image_index: number;
    width: number | null;
    height: number | null;
    description: string | null;
  };
}

export interface DemoTimeline {
  pending: boolean;
  events: DemoTimelineEvent[];
  generated_at?: string | null;
  note?: string;
}

export async function demoTimeline(): Promise<DemoTimeline> {
  return demoRequest("/api/demo/timeline");
}

export interface DemoOverview {
  pending: boolean;
  summary: string;
  parties: Array<{ role: string; name: string }>;
  key_issues: string[];
  jurisdiction?: string | null;
  matter_type?: string | null;
  generated_at?: string | null;
  note?: string;
}

export async function demoOverview(): Promise<DemoOverview> {
  return demoRequest("/api/demo/overview");
}

// ── Admin ─────────────────────────────────────────────────────────────────────

export interface AdminCosts {
  total_cost_usd: number;
  total_queries: number;
  by_user: Array<{
    user_slug: string;
    query_count: number;
    cost_usd: number;
    last_active: string | null;
    first_name: string | null;
    last_name: string | null;
    email: string | null;
    phone: string | null;
    info_submitted_at: string | null;
  }>;
  by_model: Array<{ model: string; input_tokens: number; output_tokens: number; cost_usd: number }>;
  recent_queries: Array<{ user_slug: string; timestamp: string | null; question: string; cost_usd: number }>;
}

export async function adminCosts(adminKey: string): Promise<AdminCosts> {
  return demoRequest("/api/demo/admin/costs", {
    headers: { "x-admin-key": adminKey } as Record<string, string>,
  });
}
