/**
 * Demo portal API client.
 * Uses a demo JWT cached in browser storage — no Firebase.
 */

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TOKEN_KEY = "demo_token";

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

  const res = await fetch(`${BASE}${path}`, { ...opts, headers });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json() as Promise<T>;
}

// ── Auth ──────────────────────────────────────────────────────────────────────

export async function demoAuth(accessCode: string): Promise<{ user_slug: string; token: string; ok: boolean }> {
  return demoRequest("/api/demo/auth", {
    method: "POST",
    body: JSON.stringify({ access_code: accessCode }),
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

  const res = await fetch(`${BASE}/api/demo/cases/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
  });

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
  by_user: Array<{ user_slug: string; query_count: number; cost_usd: number; last_active: string | null }>;
  by_model: Array<{ model: string; input_tokens: number; output_tokens: number; cost_usd: number }>;
  recent_queries: Array<{ user_slug: string; timestamp: string | null; question: string; cost_usd: number }>;
}

export async function adminCosts(adminKey: string): Promise<AdminCosts> {
  return demoRequest("/api/demo/admin/costs", {
    headers: { "x-admin-key": adminKey } as Record<string, string>,
  });
}
