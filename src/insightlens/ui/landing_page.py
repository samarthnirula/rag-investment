"""Landing page — shown to unauthenticated visitors."""
from __future__ import annotations

import os
import json as _json

import streamlit as st

from insightlens.ui.auth import (
    AuthError,
    send_email_verification,
    send_password_reset,
    sign_in,
    register,
)

# ── Auth helpers (unchanged) ───────────────────────────────────────────────────

def _store_user(user: dict) -> None:
    st.session_state.user_uid           = user["uid"]
    st.session_state.user_email         = user["email"]
    st.session_state.user_display_name  = user["display_name"]
    st.session_state.user_id_token      = user["id_token"]
    st.session_state.user_refresh_token = user["refresh_token"]
    st.session_state.pop("_show_auth_panel", None)


def _check_access_code(code: str, user_id: str) -> tuple[bool, str]:
    from insightlens.config import load_config
    from insightlens.storage.access_code_repository import AccessCodeRepository
    from insightlens.storage.snowflake_client import open_connection
    try:
        cfg = load_config()
        with open_connection(cfg.db) as conn:
            repo = AccessCodeRepository(conn)
            if not repo.codes_exist():
                return True, "OK"
            if not code.strip():
                return False, "An access code is required. Request one at access@atticus.ai"
            return repo.validate_and_claim(code.strip(), user_id)
    except Exception:
        return True, "OK"


def _start_public_demo() -> None:
    st.session_state.demo_mode = True
    st.session_state.active_page = "epstein"
    st.rerun()


def _firebase_redirect_bridge() -> None:
    import streamlit as _st
    if _st.query_params.get("firebase_token"):
        return

    api_key = os.getenv("FIREBASE_API_KEY") or os.getenv("FIREBASE_WEB_API_KEY", "")
    project_id = os.getenv("FIREBASE_PROJECT_ID", "")
    if not api_key or not project_id or project_id == "your-project-id-here":
        return

    auth_domain = os.getenv("FIREBASE_AUTH_DOMAIN") or f"{project_id}.firebaseapp.com"
    cfg_json = _json.dumps(
        {"apiKey": api_key, "authDomain": auth_domain, "projectId": project_id}
    )
    st.components.v1.html(f"""
    <script>
    (function(){{
      const topw = window.top;
      const now = Date.now();
      const loadingAge = now - (topw.__atticusFirebaseCompatStartedAt || 0);
      if (topw.__atticusFirebaseCompatState === 'ready' &&
          typeof topw.__atticusGoogleLogin === 'function') {{
        return;
      }}
      if (topw.__atticusFirebaseCompatState === 'loading' && loadingAge < 8000) {{
        return;
      }}
      delete topw.__atticusGoogleLogin;
      topw.__atticusFirebaseCompatState = 'loading';
      topw.__atticusFirebaseCompatStartedAt = now;

      function post(type, detail) {{
        try {{ topw.postMessage({{type, detail}}, '*'); }} catch (_) {{}}
      }}

      function loadScript(src) {{
        return new Promise(function(resolve, reject) {{
          const existing = topw.document.querySelector('script[src="' + src + '"]');
          if (existing) {{
            existing.addEventListener('load', resolve, {{once:true}});
            existing.addEventListener('error', reject, {{once:true}});
            if (existing.dataset.loaded === '1') resolve();
            return;
          }}
          const s = topw.document.createElement('script');
          s.src = src;
          s.async = true;
          s.onload = function() {{ s.dataset.loaded = '1'; resolve(); }};
          s.onerror = reject;
          topw.document.head.appendChild(s);
        }});
      }}

      function appBaseUrl() {{
        return topw.location.origin + topw.location.pathname;
      }}

      async function init() {{
        try {{
          await loadScript('https://www.gstatic.com/firebasejs/10.13.1/firebase-app-compat.js');
          await loadScript('https://www.gstatic.com/firebasejs/10.13.1/firebase-auth-compat.js');
          const firebase = topw.firebase;
          const app = firebase.apps && firebase.apps.length
            ? firebase.app()
            : firebase.initializeApp({cfg_json});
          const auth = firebase.auth(app);

          auth.getRedirectResult().then(function(result) {{
            if (!result || !result.user) return;
            if (topw.location.search.indexOf('firebase_token') !== -1) return;
            result.user.getIdToken().then(function(token) {{
              try {{
                if (result.user.refreshToken) {{
                  topw.sessionStorage.setItem('__att_rt', result.user.refreshToken);
                }}
              }} catch (_) {{}}
              const p = new URLSearchParams({{
                firebase_token: token,
                firebase_uid: result.user.uid,
                firebase_email: result.user.email || '',
                firebase_name: result.user.displayName || ''
              }});
              topw.location.href = appBaseUrl() + '?' + p.toString();
            }});
          }}).catch(function(e) {{
            topw.__atticusFirebaseCompatState = 'error';
            topw.__atticusFirebaseCompatError = e && e.code ? e.code : String(e || 'redirect-result-failed');
            post('atticus_firebase_error', topw.__atticusFirebaseCompatError);
          }});

          topw.__atticusGoogleLogin = function() {{
            if (topw.location.search.indexOf('firebase_token') !== -1) return;
            const provider = new firebase.auth.GoogleAuthProvider();
            provider.setCustomParameters({{prompt:'select_account'}});
            return auth.signInWithRedirect(provider);
          }};
          topw.__atticusFirebaseCompatState = 'ready';
          post('atticus_firebase_ready', 'ready');
        }} catch (e) {{
          topw.__atticusFirebaseCompatState = 'error';
          topw.__atticusFirebaseCompatError = e && e.message ? e.message : String(e || 'firebase-load-failed');
          post('atticus_firebase_error', topw.__atticusFirebaseCompatError);
        }}
      }}

      init();
    }})();
    </script>
    """, height=0)


def _google_signin_button(key_suffix: str = "") -> None:
    api_key    = os.getenv("FIREBASE_API_KEY") or os.getenv("FIREBASE_WEB_API_KEY", "")
    project_id = os.getenv("FIREBASE_PROJECT_ID", "")

    missing: list[str] = []
    if not api_key:
        missing.append("FIREBASE_API_KEY")
    if not project_id or project_id == "your-project-id-here":
        missing.append("FIREBASE_PROJECT_ID")

    if missing:
        missing_str = " &amp; ".join(f"<code>{v}</code>" for v in missing)
        st.html(
            f"<div style='background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;"
            f"padding:12px 16px;font-size:0.78rem;color:#991B1B;margin-bottom:4px;line-height:1.6'>"
            f"Google Sign-In needs {missing_str} in .env</div>"
        )
        return

    _firebase_redirect_bridge()

    google_svg = (
        '<svg width="18" height="18" viewBox="0 0 48 48" style="flex-shrink:0">'
        '<path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0'
        " 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>"
        '<path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94'
        "c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>"
        '<path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59'
        "l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>"
        '<path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6'
        "c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19"
        'C6.51 42.62 14.62 48 24 48z"/></svg>'
    )

    html = f"""
<style>
  #g-btn-{key_suffix} {{
    display:flex;align-items:center;justify-content:center;gap:10px;
    width:100%;padding:12px 20px;box-sizing:border-box;
    background:#FFFFFF;border:1px solid rgba(0,0,0,0.08);border-radius:12px;
    font-family:'DM Sans',sans-serif;font-size:0.88rem;font-weight:500;
    color:#2D2D3A;cursor:pointer;transition:all .2s ease;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
  }}
  #g-btn-{key_suffix}:hover  {{ box-shadow:0 4px 12px rgba(0,0,0,0.08);transform:translateY(-1px); }}
  #g-btn-{key_suffix}:disabled {{ opacity:.55;cursor:not-allowed; }}
  #g-err-{key_suffix} {{ color:#DC2626;font-size:0.78rem;margin-top:8px;text-align:center;min-height:18px; }}
</style>
<button id="g-btn-{key_suffix}">{google_svg} Continue with Google</button>
<div id="g-err-{key_suffix}"></div>
<script>
(function(){{
  const btn = document.getElementById('g-btn-{key_suffix}');
  const err = document.getElementById('g-err-{key_suffix}');
  const googleSvg = `{google_svg}`;

  window.addEventListener('message', function(e) {{
    if (!e.data || e.data.type !== 'atticus_firebase_error') return;
    err.textContent = 'Google Sign-In failed: ' + e.data.detail;
    btn.disabled = false;
    btn.innerHTML = googleSvg + ' Continue with Google';
  }});

  btn.addEventListener('click', function() {{
    btn.disabled = true;
    err.textContent = '';
    btn.innerHTML = googleSvg + ' Opening Google...';
    let tries = 0;
    const go = function() {{
      if (window.top.__atticusGoogleLogin) {{
        window.top.__atticusGoogleLogin().catch(function(e) {{
          const code = e && e.code ? e.code : String(e || 'unknown-error');
          err.textContent = 'Google Sign-In failed: ' + code;
          btn.disabled = false;
          btn.innerHTML = googleSvg + ' Continue with Google';
        }});
        return;
      }}
      if (window.top.__atticusFirebaseCompatState === 'error') {{
        err.textContent = 'Google Sign-In failed: ' +
          (window.top.__atticusFirebaseCompatError || 'Firebase failed to load');
        btn.disabled = false;
        btn.innerHTML = googleSvg + ' Continue with Google';
        return;
      }}
      if (tries++ < 40) {{
        setTimeout(go, 150);
      }} else {{
        err.textContent = 'Google Sign-In failed to load. Please refresh.';
        btn.disabled = false;
        btn.innerHTML = googleSvg + ' Continue with Google';
      }}
    }};
    go();
  }});
}})();
</script>
"""
    st.components.v1.html(html, height=92)


# ── Auth dialog ────────────────────────────────────────────────────────────────

@st.dialog("Sign in to Atticus", width="small")
def _auth_dialog() -> None:
    st.html("""
    <style>
    body { overflow: hidden !important; }
    [data-testid="stDialog"] > div {
        background: rgba(245, 244, 240, 0.98) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 20px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15) !important;
    }
    [data-testid="stDialog"] .stTextInput > div > div > input {
        background: #FFFFFF !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        color: #2D2D3A !important;
        border-radius: 10px !important;
    }
    [data-testid="stDialog"] .stTextInput > div > div > input::placeholder { color: #8A8A9A !important; }
    [data-testid="stDialog"] .stTextInput label { color: #4A4A5A !important; font-size: 0.78rem !important; }
    [data-testid="stDialog"] .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid rgba(0,0,0,0.06) !important; }
    [data-testid="stDialog"] .stTabs [data-baseweb="tab"] { color: #6B6B7A !important; font-size: 0.85rem !important; }
    [data-testid="stDialog"] .stTabs [aria-selected="true"] { color: #667EEA !important; border-bottom-color: #667EEA !important; }
    [data-testid="stDialog"] p, [data-testid="stDialog"] span { color: #4A4A5A; }
    [data-testid="stDialog"] .stCheckbox label span { color: #6B6B7A !important; font-size: 0.82rem !important; }
    [data-testid="stDialog"] .stButton > button {
        background: #FFFFFF !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        color: #4A4A5A !important;
        border-radius: 10px !important;
    }
    [data-testid="stDialog"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 12px rgba(118, 75, 162, 0.3) !important;
    }
    [data-testid="stDialog"] hr { border-color: rgba(0,0,0,0.06) !important; }
    </style>
    """)
    st.html(
        "<div style='text-align:center;padding:4px 0 20px'>"
        "<div style='display:inline-flex;align-items:center;gap:8px;margin-bottom:8px'>"
        "<div style='width:20px;height:20px;background:linear-gradient(135deg,#667EEA,#764BA2);"
        "border-radius:5px;flex-shrink:0;box-shadow:0 4px 12px rgba(118, 75, 162, 0.35)'></div>"
        "<span style='font-size:1.6rem;font-weight:700;color:#2D2D3A;"
        "letter-spacing:-0.5px'>Atticus</span>"
        "</div>"
        "<div style='font-size:0.75rem;color:#667EEA;font-weight:500;'>"
        "Legal research intelligence</div></div>"
    )

    tab_in, tab_up = st.tabs(["Sign In", "Create Account"])

    with tab_in:
        if not st.session_state.get("_dlg_reset"):
            with st.form("dlg_login"):
                email    = st.text_input("Email", placeholder="you@lawfirm.com")
                password = st.text_input("Password", type="password", placeholder="Password")
                ok = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                if ok:
                    if not email.strip() or not password:
                        st.error("Email and password are required.")
                    else:
                        try:
                            _store_user(sign_in(email.strip().lower(), password))
                            st.session_state.active_page = "insightlens"
                            st.rerun()
                        except AuthError as exc:
                            st.error(str(exc))
                        except Exception:
                            st.error("Connection error. Please try again.")
            if st.button("Forgot password?", use_container_width=True, key="dlg_forgot"):
                st.session_state._dlg_reset = True
                st.rerun()
        else:
            st.html("<div style='font-size:0.88rem;font-weight:600;color:#2D2D3A;margin-bottom:10px'>Reset password</div>")
            re_email = st.text_input("Email", placeholder="you@lawfirm.com", key="dlg_reset_email")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Send link", type="primary", use_container_width=True, key="dlg_send_reset"):
                    if re_email.strip():
                        try:
                            send_password_reset(re_email.strip().lower())
                            st.success("Reset link sent.")
                            st.session_state._dlg_reset = False
                        except AuthError as exc:
                            st.error(str(exc))
                    else:
                        st.error("Enter your email.")
            with c2:
                if st.button("Back", use_container_width=True, key="dlg_reset_back"):
                    st.session_state._dlg_reset = False
                    st.rerun()

    with tab_up:
        st.html(
            "<div style='font-size:0.75rem;color:#6B6B7A;margin-bottom:12px;line-height:1.6'>"
            "<strong style='color:#4A4A5A'>Create your account with email.</strong><br>"
            "Google sign-in is also available below if your domain allows it.</div>"
        )
        with st.form("dlg_register"):
            r_email = st.text_input("Work email", placeholder="you@lawfirm.com")
            r_pass  = st.text_input("Password", type="password", placeholder="Min 8 characters")
            r_conf  = st.text_input("Confirm password", type="password", placeholder="Confirm password")
            r_code  = st.text_input("Access code", placeholder="Optional during pilot")
            accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            ok = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if ok:
                if not r_email.strip() or not r_pass or not r_conf:
                    st.error("Email and password are required.")
                elif r_pass != r_conf:
                    st.error("Passwords do not match.")
                elif len(r_pass) < 8:
                    st.error("Password must be at least 8 characters.")
                elif not accepted:
                    st.error("You must agree to the Terms of Service.")
                else:
                    try:
                        user = register(r_email.strip().lower(), r_pass)
                        ok_code, msg = _check_access_code(r_code, user["uid"])
                        if not ok_code:
                            st.error(msg)
                        else:
                            _store_user(user)
                            try:
                                send_email_verification(user["id_token"])
                            except Exception:
                                pass
                            st.session_state._email_verified = False
                            st.session_state.active_page = "insightlens"
                            st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Sign-up failed: {exc}")

    st.html(
        "<div style='display:flex;align-items:center;gap:12px;margin:18px 0 14px'>"
        "<div style='flex:1;height:1px;background:rgba(0,0,0,0.08)'></div>"
        "<span style='font-size:0.72rem;color:#8A8A9A'>or continue with Google</span>"
        "<div style='flex:1;height:1px;background:rgba(0,0,0,0.08)'></div>"
        "</div>"
    )
    _google_signin_button()

    st.html(
        "<div style='text-align:center;font-size:0.68rem;color:#8A8A9A;"
        "margin-top:16px;padding-top:12px;line-height:1.7'>"
        "AI responses are research aids only, not legal advice."
        "</div>"
    )


# ── Landing page ───────────────────────────────────────────────────────────────

def _render_auth_panel() -> None:
    """Inline sign-in / register panel — rendered in the main script context."""
    st.html("""
    <style>
    .att-auth-panel .stTextInput > div > div > input {
        background: #FFFFFF !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        color: #2D2D3A !important;
        border-radius: 10px !important;
    }
    .att-auth-panel .stTextInput > div > div > input::placeholder { color: #8A8A9A !important; }
    .att-auth-panel .stTextInput label { color: #4A4A5A !important; font-size: 0.78rem !important; }
    .att-auth-panel .stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid rgba(0,0,0,0.06) !important; }
    .att-auth-panel .stTabs [data-baseweb="tab"] { color: #6B6B7A !important; font-size: 0.85rem !important; }
    .att-auth-panel .stTabs [aria-selected="true"] { color: #667EEA !important; border-bottom-color: #667EEA !important; }
    .att-auth-panel p, .att-auth-panel span { color: #4A4A5A; }
    .att-auth-panel .stCheckbox label span { color: #6B6B7A !important; font-size: 0.82rem !important; }
    .att-auth-panel .stButton > button {
        background: #FFFFFF !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        color: #4A4A5A !important;
        border-radius: 10px !important;
    }
    .att-auth-panel .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 12px rgba(118, 75, 162, 0.3) !important;
    }
    </style>
    """)

    _, panel_col, _ = st.columns([1, 2, 1])
    with panel_col:
        st.html(
            "<div class='att-auth-panel' style='background:#FFFFFF;"
            "border:1px solid rgba(0,0,0,0.08);border-radius:20px;padding:32px 28px 24px;"
            "box-shadow:0 20px 60px rgba(0,0,0,0.12);margin-bottom:24px;max-width:420px;margin:0 auto'>"
        )
        st.html(
            "<div style='text-align:center;padding:4px 0 20px'>"
            "<div style='display:inline-flex;align-items:center;gap:8px;margin-bottom:8px'>"
            "<div style='width:20px;height:20px;background:linear-gradient(135deg,#667EEA,#764BA2);"
            "border-radius:5px;flex-shrink:0;box-shadow:0 4px 12px rgba(118, 75, 162, 0.35)'></div>"
            "<span style='font-size:1.6rem;font-weight:700;color:#2D2D3A;"
            "letter-spacing:-0.5px'>Atticus</span></div>"
            "<div style='font-size:0.75rem;color:#667EEA;font-weight:500'>Legal research intelligence</div></div>"
        )

        tab_in, tab_up = st.tabs(["Sign In", "Create Account"])

        with tab_in:
            if not st.session_state.get("_panel_reset"):
                with st.form("panel_login"):
                    email    = st.text_input("Email", placeholder="you@lawfirm.com", key="panel_email")
                    password = st.text_input("Password", type="password", placeholder="Password", key="panel_pass")
                    ok = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                    if ok:
                        if not email.strip() or not password:
                            st.error("Email and password are required.")
                        else:
                            try:
                                _store_user(sign_in(email.strip().lower(), password))
                                st.session_state.active_page = "insightlens"
                                st.rerun()
                            except AuthError as exc:
                                st.error(str(exc))
                            except Exception:
                                st.error("Connection error. Please try again.")
                if st.button("Forgot password?", use_container_width=True, key="panel_forgot"):
                    st.session_state._panel_reset = True
                    st.rerun()
            else:
                re_email = st.text_input("Email for reset", placeholder="you@lawfirm.com", key="panel_reset_email")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Send link", type="primary", use_container_width=True, key="panel_send_reset"):
                        if re_email.strip():
                            try:
                                send_password_reset(re_email.strip().lower())
                                st.success("Reset link sent.")
                                st.session_state._panel_reset = False
                            except AuthError as exc:
                                st.error(str(exc))
                        else:
                            st.error("Enter your email.")
                with c2:
                    if st.button("Back", use_container_width=True, key="panel_reset_back"):
                        st.session_state._panel_reset = False
                        st.rerun()

        with tab_up:
            with st.form("panel_register"):
                r_email = st.text_input("Work email", placeholder="you@lawfirm.com", key="panel_reg_email")
                r_pass  = st.text_input("Password", type="password", placeholder="Min 8 characters", key="panel_reg_pass")
                r_conf  = st.text_input("Confirm password", type="password", key="panel_reg_conf")
                r_code  = st.text_input("Access code", placeholder="Optional during pilot", key="panel_reg_code")
                accepted = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="panel_tos")
                ok = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if ok:
                    if not r_email.strip() or not r_pass or not r_conf:
                        st.error("Email and password are required.")
                    elif r_pass != r_conf:
                        st.error("Passwords do not match.")
                    elif len(r_pass) < 8:
                        st.error("Password must be at least 8 characters.")
                    elif not accepted:
                        st.error("You must agree to the Terms of Service.")
                    else:
                        try:
                            user = register(r_email.strip().lower(), r_pass)
                            ok_code, msg = _check_access_code(r_code, user["uid"])
                            if not ok_code:
                                st.error(msg)
                            else:
                                _store_user(user)
                                try:
                                    send_email_verification(user["id_token"])
                                except Exception:
                                    pass
                                st.session_state._email_verified = False
                                st.session_state.active_page = "insightlens"
                                st.rerun()
                        except AuthError as exc:
                            st.error(str(exc))
                        except Exception as exc:
                            st.error(f"Sign-up failed: {exc}")

        _firebase_redirect_bridge()
        st.html(
            "<div style='display:flex;align-items:center;gap:12px;margin:18px 0 14px'>"
            "<div style='flex:1;height:1px;background:rgba(0,0,0,0.08)'></div>"
            "<span style='font-size:0.72rem;color:#8A8A9A'>or continue with Google</span>"
            "<div style='flex:1;height:1px;background:rgba(0,0,0,0.08)'></div></div>"
        )
        _google_signin_button(key_suffix="panel")

        if st.button("Close", key="panel_close", use_container_width=True):
            st.session_state._show_auth_panel = False
            st.rerun()

        st.html("</div>")


def render_landing_page() -> None:
    if st.session_state.get("google_auth_error"):
        st.error(st.session_state.pop("google_auth_error"))

    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body { background: #F5F4F0 !important; overflow: hidden; }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main,
    [data-testid="stAppViewContainer"] > .main,
    .block-container,
    [data-testid="stHorizontalBlock"],
    [data-testid="stVerticalBlock"],
    [data-testid="column"] { background: #F5F4F0 !important; }

    [data-testid="stAppViewContainer"] > .main,
    .block-container { padding:0!important; max-width:100%!important; }
    [data-testid="stSidebar"],
    footer, #MainMenu,
    header[data-testid="stHeader"] { display:none!important; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif!important; }

    .stButton > button {
        font-family: 'DM Sans', sans-serif !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        color: #4A4A5A !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #FAFAF8 !important;
        border-color: rgba(102, 126, 234, 0.3) !important;
        color: #667EEA !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 4px 16px rgba(118, 75, 162, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5B71E8 0%, #6A4199 100%) !important;
        box-shadow: 0 6px 20px rgba(118, 75, 162, 0.45) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="tertiary"] {
        background: transparent !important;
        border: none !important;
        color: #6B6B7A !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        transition: all 0.15s !important;
    }
    .stButton > button[kind="tertiary"]:hover {
        background: rgba(102, 126, 234, 0.08) !important;
        border-radius: 8px !important;
        color: #667EEA !important;
    }

    @media (max-width: 640px) {
        [data-testid="stButton"]:has(button[kind="tertiary"]) { display: none !important; }
    }
    </style>
    """)

    # ── Auth overlay (when panel is open, cover everything) ─────────────────────
    if st.session_state.get("_show_auth_panel"):
        st.components.v1.html("""
        <style>
        body { overflow: hidden !important; }
        </style>
        """, height=0)
        _render_auth_panel()
        return

    # ── NAV ───────────────────────────────────────────────────────────────────
    nav_l, _, nav_a, nav_t, nav_p, nav_r = st.columns([3, 2, 1, 1, 1, 0.8])
    with nav_l:
        st.html(
            "<div style='padding:20px 0 16px 40px;display:flex;align-items:center;gap:10px'>"
            "<div style='width:22px;height:22px;background:linear-gradient(135deg,#667EEA,#764BA2);"
            "border-radius:5px;flex-shrink:0;box-shadow:0 4px 12px rgba(118, 75, 162, 0.35)'></div>"
            "<span style='font-size:1.15rem;font-weight:700;color:#2D2D3A;"
            "letter-spacing:-0.3px'>Atticus</span>"
            "<span style='font-size:0.68rem;color:#667EEA;margin-left:6px;font-weight:500'>"
            "Legal Research Intelligence</span>"
            "</div>"
        )
    with nav_a:
        if st.button("About", key="lp_nav_about", type="tertiary", use_container_width=True):
            st.session_state.pre_auth_page = "about"
            st.rerun()
    with nav_t:
        if st.button("Terms", key="lp_nav_terms", type="tertiary", use_container_width=True):
            st.session_state.pre_auth_page = "terms"
            st.rerun()
    with nav_p:
        if st.button("Privacy", key="lp_nav_privacy", type="tertiary", use_container_width=True):
            st.session_state.pre_auth_page = "privacy"
            st.rerun()
    with nav_r:
        if st.button("Sign in", key="lp_nav_login", type="secondary", use_container_width=True):
            st.session_state._show_auth_panel = True
            st.rerun()

    # ── HERO ──────────────────────────────────────────────────────────────────
    st.html(
        "<div style='background:linear-gradient(135deg,#FFFFFF 0%,#F8F7F4 100%);"
        "position:relative;overflow:hidden'>"
        "<div style='position:absolute;inset:0;"
        "background:radial-gradient(circle at 70% 30%,rgba(102,126,234,0.06) 0%,transparent 50%),"
        "radial-gradient(circle at 30% 70%,rgba(118,75,162,0.04) 0%,transparent 50%)'></div>"
        "</div>"
    )

    h_left, h_right = st.columns([1, 1], gap="large")

    with h_left:
        st.html(
            "<div style='padding:80px 0 80px 48px;position:relative;z-index:1'>"
            "<div style='display:inline-flex;align-items:center;gap:6px;"
            "background:linear-gradient(135deg,rgba(102,126,234,0.1),rgba(118,75,162,0.08));"
            "border:1px solid rgba(102,126,234,0.2);"
            "color:#667EEA;font-size:0.72rem;font-weight:600;letter-spacing:0.05em;"
            "border-radius:99px;padding:6px 14px;margin-bottom:28px'>"
            "Premium Legal AI - Early Access"
            "</div>"
            "<h1 style='font-size:3.4rem;font-weight:800;color:#2D2D3A;"
            "line-height:1.08;letter-spacing:-1.5px;margin:0 0 24px'>"
            "Find the facts.<br>"
            "<span style='background:linear-gradient(90deg,#667EEA,#764BA2);"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "background-clip:text'>Every answer cited.</span>"
            "</h1>"
            "<p style='font-size:1.08rem;color:#6B6B7A;line-height:1.75;"
            "margin:0 0 40px;max-width:480px'>"
            "Atticus reads your case files, flags risk clauses, and answers legal questions "
            "with exact page citations. Built for attorneys who can't afford to miss a detail."
            "</p>"
            "</div>"
        )
        cta1, _ = st.columns([2, 5])
        with cta1:
            if st.button("Try Public Demo", key="lp_hero_demo", type="primary",
                         use_container_width=True):
                _start_public_demo()

    with h_right:
        st.html(
            "<div style='padding:60px 48px 60px 16px;position:relative;z-index:1'>"
            "<div style='"
            "background:#FFFFFF;border:1px solid rgba(0,0,0,0.06);"
            "border-radius:16px;overflow:hidden;"
            "box-shadow:0 8px 40px rgba(0,0,0,0.08)'>"
            "<div style='background:linear-gradient(135deg,#F8F7F4,#FFFFFF);border-bottom:1px solid rgba(0,0,0,0.04);"
            "padding:12px 16px;display:flex;align-items:center;gap:8px'>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#667EEA;opacity:.7'></div>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#764BA2;opacity:.7'></div>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#A78BFA;opacity:.7'></div>"
            "<span style='margin-left:10px;font-size:0.72rem;color:#6B6B7A'>Atticus - Contract Review</span>"
            "</div>"
            "<div style='padding:16px;border-bottom:1px solid rgba(0,0,0,0.04)'>"
            "<div style='display:flex;justify-content:flex-end'>"
            "<div style='background:linear-gradient(135deg,#667EEA,#764BA2);border-radius:14px 14px 4px 14px;"
            "padding:10px 14px;max-width:80%;font-size:0.8rem;color:#FFFFFF;line-height:1.55'>"
            "Review the NDA and flag any clauses that could harm my client."
            "</div></div></div>"
            "<div style='padding:16px'>"
            "<div style='display:flex;align-items:center;gap:6px;margin-bottom:12px;"
            "padding:8px 12px;background:rgba(102,126,234,0.06);border-radius:8px;"
            "font-size:0.72rem;color:#667EEA'>"
            "Document - Harmon_NDA_v3.pdf - 14 pages"
            "</div>"
            "<div style='font-size:0.88rem;font-weight:500;color:#2D2D3A;margin-bottom:14px;line-height:1.6'>"
            "Found 3 high-risk and 2 moderate-risk clauses. Liability and IP sections need immediate attention."
            "</div>"
            "<div style='font-size:0.65rem;font-weight:700;color:#8A8A9A;"
            "letter-spacing:.08em;margin-bottom:8px'>RISK FLAGS</div>"
            "<div style='display:flex;align-items:flex-start;gap:10px;"
            "padding:10px 12px;border:1px solid rgba(220,38,38,0.15);border-radius:10px;"
            "background:rgba(220,38,38,0.04);margin-bottom:6px'>"
            "<span style='width:8px;height:8px;border-radius:50%;background:#DC2626;"
            "flex-shrink:0;margin-top:4px'></span>"
            "<span style='flex:1;font-size:0.8rem;color:#2D2D3A;line-height:1.5'>"
            "Overly broad IP assignment: includes pre-existing work</span>"
            "<span style='font-size:0.72rem;color:#6B6B7A;white-space:nowrap'>4.2</span>"
            "<span style='background:rgba(102,126,234,0.1);color:#667EEA;font-size:0.65rem;font-weight:600;"
            "padding:3px 8px;border-radius:99px'>p.6</span>"
            "</div>"
            "<div style='display:flex;align-items:flex-start;gap:10px;"
            "padding:10px 12px;border:1px solid rgba(220,38,38,0.15);border-radius:10px;"
            "background:rgba(220,38,38,0.04);margin-bottom:6px'>"
            "<span style='width:8px;height:8px;border-radius:50%;background:#DC2626;"
            "flex-shrink:0;margin-top:4px'></span>"
            "<span style='flex:1;font-size:0.8rem;color:#2D2D3A;line-height:1.5'>"
            "Non-compete extends 36 months, no geographic limit</span>"
            "<span style='font-size:0.72rem;color:#6B6B7A;white-space:nowrap'>7.1</span>"
            "<span style='background:rgba(102,126,234,0.1);color:#667EEA;font-size:0.65rem;font-weight:600;"
            "padding:3px 8px;border-radius:99px'>p.10</span>"
            "</div>"
            "<div style='display:flex;align-items:flex-start;gap:10px;"
            "padding:10px 12px;border:1px solid rgba(217,119,6,0.15);border-radius:10px;"
            "background:rgba(217,119,6,0.04);margin-bottom:12px'>"
            "<span style='width:8px;height:8px;border-radius:50%;background:#D97706;"
            "flex-shrink:0;margin-top:4px'></span>"
            "<span style='flex:1;font-size:0.8rem;color:#2D2D3A;line-height:1.5'>"
            "Liability cap asymmetric: caps client, not counterparty</span>"
            "<span style='font-size:0.72rem;color:#6B6B7A;white-space:nowrap'>9.3</span>"
            "<span style='background:rgba(102,126,234,0.1);color:#667EEA;font-size:0.65rem;font-weight:600;"
            "padding:3px 8px;border-radius:99px'>p.12</span>"
            "</div>"
            "<div style='display:flex;gap:8px;flex-wrap:wrap'>"
            "<span style='background:linear-gradient(135deg,rgba(102,126,234,0.1),rgba(118,75,162,0.08));"
            "border:1px solid rgba(102,126,234,0.2);"
            "color:#667EEA;font-size:0.72rem;padding:6px 12px;border-radius:8px'>"
            "Draft redline 7.1</span>"
            "<span style='background:linear-gradient(135deg,rgba(102,126,234,0.1),rgba(118,75,162,0.08));"
            "border:1px solid rgba(102,126,234,0.2);"
            "color:#667EEA;font-size:0.72rem;padding:6px 12px;border-radius:8px'>"
            "Explain 4.2 risk</span>"
            "</div>"
            "</div></div></div>"
        )

    st.html("<div style='background:#F5F4F0;height:48px'></div>")

    # ── STATS BAR ─────────────────────────────────────────────────────────────
    st.html(
        "<div style='background:#FFFFFF;border-top:1px solid rgba(0,0,0,0.04);"
        "border-bottom:1px solid rgba(0,0,0,0.04);padding:32px 48px'>"
        "<div style='display:flex;justify-content:space-around;flex-wrap:wrap;gap:24px'>"
        + "".join(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.9rem;font-weight:800;color:{c};letter-spacing:-0.5px'>{v}</div>"
            f"<div style='font-size:0.72rem;color:#8A8A9A;margin-top:4px'>{l}</div>"
            f"</div>"
            for v, l, c in [
                ("Demo", "Public access first", "#667EEA"),
                ("100%", "Your docs, your answers", "#16A34A"),
                ("8-stage", "Hybrid retrieval pipeline", "#764BA2"),
                ("GDPR", "Data deleted on request", "#059669"),
            ]
        )
        + "</div></div>"
    )

    # ── FEATURES ──────────────────────────────────────────────────────────────
    st.html(
        "<div style='background:#F5F4F0;padding:80px 48px'>"
        "<div style='text-align:center;margin-bottom:48px'>"
        "<div style='font-size:0.68rem;font-weight:700;color:#667EEA;"
        "letter-spacing:.1em;margin-bottom:14px'>WHAT YOU GET</div>"
        "<h2 style='font-size:2.2rem;font-weight:800;color:#2D2D3A;"
        "line-height:1.12;letter-spacing:-1px;margin:0'>"
        "Everything a litigator needs.</h2>"
        "</div></div>"
    )

    fc1, fc2, fc3, fc4 = st.columns(4, gap="medium")
    _feats = [
        ("Case Chat",
         "Ask questions in plain English. Get cited answers from the exact pages of your documents."),
        ("Risk Flags",
         "Automatic risk scoring for contracts. High, medium, low ratings with clause and page references."),
        ("Case Timeline",
         "Auto-structured visual timelines from your case files. Filter by event type instantly."),
        ("Team Forum",
         "Per-case discussion board for your team. Post findings, export as a markdown report."),
    ]
    for col, (title, body) in zip([fc1, fc2, fc3, fc4], _feats):
        with col:
            st.html(
                f"<div style='background:#FFFFFF;border:1px solid rgba(0,0,0,0.06);"
                f"border-radius:16px;padding:24px 20px;height:100%;box-shadow:0 2px 8px rgba(0,0,0,0.03)'>"
                f"<div style='font-size:1rem;font-weight:700;color:#2D2D3A;margin-bottom:12px'>{title}</div>"
                f"<div style='font-size:0.82rem;color:#6B6B7A;line-height:1.65'>{body}</div>"
                f"</div>"
            )

    # ── HOW IT WORKS ──────────────────────────────────────────────────────────
    st.html(
        "<div style='background:#FFFFFF;border-top:1px solid rgba(0,0,0,0.04);"
        "border-bottom:1px solid rgba(0,0,0,0.04);padding:80px 48px'>"
        "<div style='text-align:center;margin-bottom:52px'>"
        "<div style='font-size:0.68rem;font-weight:700;color:#667EEA;"
        "letter-spacing:.1em;margin-bottom:14px'>HOW IT WORKS</div>"
        "<h2 style='font-size:2.2rem;font-weight:800;color:#2D2D3A;"
        "line-height:1.12;letter-spacing:-1px;margin:0'>"
        "Three steps. Seconds to your first answer.</h2>"
        "</div>"
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:48px;max-width:880px;margin:0 auto'>"
        + "".join(
            f"<div style='text-align:center'>"
            f"<div style='width:48px;height:48px;border-radius:12px;"
            f"background:linear-gradient(135deg,rgba(102,126,234,0.12),rgba(118,75,162,0.1));"
            f"border:1px solid rgba(102,126,234,0.2);"
            f"color:#667EEA;font-size:1.2rem;font-weight:700;"
            f"display:flex;align-items:center;justify-content:center;margin:0 auto 18px'>{n}</div>"
            f"<div style='font-size:1rem;font-weight:600;color:#2D2D3A;margin-bottom:10px'>{t}</div>"
            f"<div style='font-size:0.82rem;color:#6B6B7A;line-height:1.65'>{d}</div>"
            f"</div>"
            for n, t, d in [
                ("1", "Upload your files",
                 "PDF, scanned or digital. OCR handles everything. Text, tables, images, footnotes."),
                ("2", "Ask any question",
                 "Plain English. No query language. 'What did the defendant say about wire transfers?'"),
                ("3", "Get cited answers",
                 "Every answer cites the exact source page. Click to expand. Export as markdown."),
            ]
        )
        + "</div></div>"
    )

    # ── PRICING ───────────────────────────────────────────────────────────────
    st.html(
        "<div style='background:#F5F4F0;padding:80px 48px'>"
        "<div style='text-align:center;margin-bottom:48px'>"
        "<div style='font-size:0.68rem;font-weight:700;color:#667EEA;"
        "letter-spacing:.1em;margin-bottom:14px'>PRICING</div>"
        "<h2 style='font-size:2.2rem;font-weight:800;color:#2D2D3A;"
        "line-height:1.12;letter-spacing:-1px;margin:0'>Simple. No surprises.</h2>"
        "</div>"
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);"
        "gap:24px;max-width:900px;margin:0 auto'>"
        + "".join(
            f"<div style='background:{bg};border:{brd};border-radius:20px;"
            f"padding:32px 28px;position:relative'>"
            + (f"<div style='position:absolute;top:-14px;left:50%;"
               f"transform:translateX(-50%);background:linear-gradient(135deg,#667EEA,#764BA2);color:#fff;"
               f"font-size:0.65rem;font-weight:700;padding:4px 16px;"
               f"border-radius:99px;white-space:nowrap'>MOST POPULAR</div>" if pop else "")
            + f"<div style='font-size:0.68rem;font-weight:700;color:#667EEA;"
            f"letter-spacing:.08em;margin-bottom:14px'>{tier}</div>"
            f"<div style='font-size:2.2rem;font-weight:800;color:#2D2D3A;margin-bottom:6px'>{price}</div>"
            f"<div style='font-size:0.72rem;color:#8A8A9A;margin-bottom:24px'>{sub}</div>"
            f"<ul style='list-style:none;padding:0;margin:0;"
            f"font-size:0.82rem;color:#4A4A5A;line-height:2.2'>"
            + "".join(f"<li>- {f}</li>" for f in fs)
            + f"</ul></div>"
            for tier, price, sub, fs, bg, brd, pop in [
                ("SOLO", "$79", "/month",
                 ["Unlimited uploads", "500 AI queries/mo", "Case Board & Timeline", "PDF + OCR"],
                 "#FFFFFF", "1px solid rgba(0,0,0,0.06)", False),
                ("FIRM", "$149", "/seat/month",
                 ["Everything in Solo", "Shared Case Boards", "Team Discussion", "Priority support", "Audit export"],
                 "linear-gradient(135deg,rgba(102,126,234,0.08),rgba(118,75,162,0.06))", "1px solid rgba(102,126,234,0.3)", True),
                ("ENTERPRISE", "Custom", "",
                 ["Everything in Firm", "SSO / SAML", "Custom retention", "Onboarding", "SLA & DPA"],
                 "#FFFFFF", "1px solid rgba(0,0,0,0.06)", False),
            ]
        )
        + "</div></div>"
    )

    # ── FINAL CTA ─────────────────────────────────────────────────────────────
    st.html(
        "<div style='background:#FFFFFF;border-top:1px solid rgba(0,0,0,0.04);"
        "padding:80px 48px;text-align:center'>"
        "<h2 style='font-size:2.4rem;font-weight:800;color:#2D2D3A;"
        "line-height:1.12;letter-spacing:-1px;margin:0 0 16px'>"
        "Every case deserves this level of research.</h2>"
        "<p style='font-size:0.98rem;color:#6B6B7A;line-height:1.7;margin:0 0 40px'>"
        "Try the public demo before you create an account.</p>"
        "</div>"
    )
    _, fcta, _ = st.columns([1, 2, 1])
    with fcta:
        if st.button("Try public demo", key="lp_final_demo",
                     type="primary", use_container_width=True):
            _start_public_demo()

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.html(
        "<div style='background:#F5F4F0;border-top:1px solid rgba(0,0,0,0.04);"
        "padding:28px 48px;display:flex;justify-content:space-between;"
        "align-items:center;flex-wrap:wrap;gap:16px'>"
        "<span style='font-size:0.78rem;color:#8A8A9A'>"
        "<strong style='color:#4A4A5A'>Atticus</strong> - 2026</span>"
        "<span style='font-size:0.72rem;color:#8A8A9A'>"
        "AI responses are research aids only, not legal advice.</span>"
        "<div style='display:flex;gap:20px'>"
        "<a style='font-size:0.72rem;color:#6B6B7A;text-decoration:none' href='#'>Terms</a>"
        "<a style='font-size:0.72rem;color:#6B6B7A;text-decoration:none' href='#'>Privacy</a>"
        "<a style='font-size:0.72rem;color:#6B6B7A;text-decoration:none' href='#'>Security</a>"
        "</div>"
        "</div>"
    )