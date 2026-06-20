# Migration Verification Checklist
**From:** Streamlit + Firebase Auth (loop) → Next.js + FastAPI + Firebase Admin SDK

---

## 1. Firebase Configuration

- [ ] Firebase project has **Email/Password** auth method enabled in Firebase Console
- [ ] Firebase project has **Google** auth method enabled in Firebase Console
- [ ] `frontend/.env.local` exists with all `NEXT_PUBLIC_FIREBASE_*` variables from Firebase Console
- [ ] `FIREBASE_SERVICE_ACCOUNT_PATH` in `.env` points to a valid Firebase Admin service account JSON
  - Or: `GOOGLE_APPLICATION_CREDENTIALS` set if using GCP default credentials
- [ ] Firebase allowlist contains `http://localhost:3000` (and production origin) in Firebase Console

**Verify:** Run `cd frontend && npm run dev`, open `http://localhost:3000`, click Sign In — auth panel should appear with email/password and Google options.

---

## 2. FastAPI Backend Startup

- [ ] `cd backend && python main.py` starts without import errors
- [ ] `GET /api/health` returns `{"status": "ok", "service": "atticus-api"}`
- [ ] Backend connects to PostgreSQL (check logs for `load_config()` success)
- [ ] `FIREBASE_SERVICE_ACCOUNT_PATH` env var resolves to a real JSON file

**Verify:** `curl http://localhost:8000/api/health`

---

## 3. Firebase Auth — Server-Side Token Verification

- [ ] `POST /api/auth/session` with a valid Firebase ID token returns user info
- [ ] `POST /api/auth/session` without a token returns `401`
- [ ] `POST /api/auth/session` with an expired/invalid token returns `401`
- [ ] `POST /api/auth/refresh` successfully exchanges a refresh token for new ID token

**Verify:** Log in via the browser dev tools → Network tab → find a request with `Authorization: Bearer <token>` → check response is 200.

---

## 4. Chat Persistence (data pipeline — no changes expected)

- [ ] `GET /api/chats` returns existing chats from PostgreSQL (no data lost)
- [ ] `POST /api/chats` creates a new chat row in `chat_summaries` table
- [ ] `GET /api/chats/{id}/messages` returns persisted messages
- [ ] `POST /api/chats/{id}/messages` saves a new message
- [ ] `DELETE /api/chats/{id}` removes the chat and its messages

**Verify:** Create a chat in the UI → reload the page → chat should still appear.

---

## 5. Query Pipeline (data pipeline — no changes expected)

- [ ] `POST /api/query` with a valid query returns an answer with sources
- [ ] Audit record is written to `audit_log` table (check `queries_today` increments)
- [ ] `GET /api/usage` returns correct query counts
- [ ] Company filter in `POST /api/query` correctly scopes results

**Verify:** Send a query → check `audit_log` table has a new row with your UID and the query text.

---

## 6. Next.js Frontend — Route Structure

- [ ] `/` renders the landing page with nav, hero, pricing sections
- [ ] `/demo` renders the public demo page (no auth required)
- [ ] `/chat` redirects to `/` when unauthenticated
- [ ] `/chat` renders the chat interface with sidebar when authenticated
- [ ] `/cases`, `/data`, `/profile` redirect to `/` when unauthenticated

**Verify:** Open each route in an incognito window (no auth) — should see landing or demo pages, not broken errors.

---

## 7. Firebase Auth — Browser-Side (OAuth Redirect Flow)

- [ ] Sign in with Google → redirects to Google → returns to the app → user is authenticated
- [ ] Sign in with email/password → user is authenticated without redirect issues
- [ ] `AuthContext.user` is populated on all authenticated pages after sign-in
- [ ] `idToken` is attached to all API requests (check Network tab → headers)
- [ ] Sign out clears the user and redirects to `/`

**Verify:** Open browser DevTools → Network → filter by `localhost:8000` → confirm every API request has `Authorization: Bearer <token>` header.

---

## 8. Docker Deployment

- [ ] `docker-compose build` completes without errors
- [ ] `docker-compose up` starts backend on port 8000 and frontend on port 3000
- [ ] Health checks pass for both services
- [ ] `firebase-service-account.json` is copied into the backend container at `/app/`
- [ ] `FIREBASE_SERVICE_ACCOUNT_PATH=/app/firebase-service-account.json` is set in docker-compose

**Verify:** `docker ps` shows both containers running → `curl http://localhost:3000` returns the Next.js app → `curl http://localhost:8000/api/health` returns `{"status": "ok"}`

---

## 9. Data Pipeline Integrity (no changes expected)

- [ ] Existing PDFs in `data/raw_pdfs/` are still indexed and queryable
- [ ] `scripts/ingest_documents.py` and `scripts/ingest_epstein.py` still work as before
- [ ] No PostgreSQL schema changes — all existing tables (`chat_summaries`, `messages`, `audit_log`, etc.) are intact
- [ ] No Snowflake dependencies remain in the API path (all data flows through PostgreSQL via existing repos)

**Verify:** Run a query that was known to work before migration — answer and citations should be identical.

---

## 10. Rollback Plan

If anything goes wrong:
1. `git checkout HEAD -- src/` — restore Streamlit app (keep `frontend/` and `backend/` as separate worktree)
2. Revert `docker-compose.yml` to the previous version
3. Firebase auth loop issue: re-apply the `st.session_state.pop("_show_auth_panel")` fix in `src/insightlens/ui/landing_page.py`

---

## Quick Smoke Test (run after setup)

```bash
# 1. Start backend
cd backend && python main.py &
sleep 3
curl http://localhost:8000/api/health

# 2. Start frontend
cd frontend && npm run dev &
# open http://localhost:3000

# 3. Sign in (email/password or Google)
# Check browser DevTools → Network → /api/* requests have Authorization header

# 4. Create a chat and send a query
# Check audit_log table: SELECT * FROM audit_log ORDER BY created_at DESC LIMIT 1;

# 5. Tear down
pkill -f "uvicorn backend.main"
pkill -f "next start"
```