"""Organization / team management page."""
from __future__ import annotations

import html

import streamlit as st

from insightlens.config import AppConfig
from insightlens.storage.org_repository import OrgRepository
from insightlens.storage.snowflake_client import open_connection

_ROLE_BADGE = {
    "owner": "<span style='background:#fef9c3;color:#854d0e;font-size:0.68rem;"
              "font-weight:700;padding:2px 8px;border-radius:99px'>OWNER</span>",
    "admin": "<span style='background:#ede9fe;color:#4f46e5;font-size:0.68rem;"
              "font-weight:700;padding:2px 8px;border-radius:99px'>ADMIN</span>",
    "member": "<span style='background:#f1f5f9;color:#475569;font-size:0.68rem;"
               "font-weight:600;padding:2px 8px;border-radius:99px'>MEMBER</span>",
}


def render_org_page(cfg: AppConfig, user: dict | None) -> None:
    if not user:
        st.info("Sign in to manage your team.")
        return

    uid = user.get("localId") or user.get("uid") or ""

    st.html(
        "<div style='font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:4px'>"
        "Team & Organization</div>"
        "<div style='font-size:0.82rem;color:#64748b;margin-bottom:20px'>"
        "Create a team workspace to share documents and cases with colleagues.</div>"
    )

    with open_connection(cfg.db) as conn:
        repo = OrgRepository(conn)
        orgs = repo.list_user_orgs(uid)

    # ── Create new org ─────────────────────────────────────────────────────────
    with st.expander("➕ Create new organization", expanded=not orgs):
        with st.form("create_org_form"):
            org_name = st.text_input("Organization name", placeholder="Acme Legal Group")
            submitted = st.form_submit_button("Create", type="primary")
            if submitted:
                name = org_name.strip()
                if not name:
                    st.error("Enter an organization name.")
                elif len(name) > 200:
                    st.error("Name must be under 200 characters.")
                else:
                    with open_connection(cfg.db) as conn:
                        OrgRepository(conn).create_org(uid, name)
                    st.success(f"Organization '{name}' created.")
                    st.rerun()

    if not orgs:
        st.caption("You are not a member of any organization yet.")
        return

    # ── Org switcher ───────────────────────────────────────────────────────────
    org_map = {o.org_name: o for o in orgs}
    selected_name = st.selectbox(
        "Active organization",
        list(org_map.keys()),
        key="active_org_select",
    )
    org = org_map[selected_name]
    st.session_state["active_org_id"] = org.org_id

    st.html(
        f"<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
        f"letter-spacing:0.06em;margin:20px 0 10px'>MEMBERS</div>"
    )

    with open_connection(cfg.db) as conn:
        members = OrgRepository(conn).list_members(org.org_id)

    is_owner = org.owner_id == uid
    is_admin = is_owner or any(
        m["user_id"] == uid and m["role"] in ("owner", "admin") for m in members
    )

    for member in members:
        mid = member["user_id"]
        role = member["role"]
        badge = _ROLE_BADGE.get(role, _ROLE_BADGE["member"])
        joined = member["joined_at"].strftime("%b %d, %Y") if member["joined_at"] else ""

        col_id, col_role, col_action = st.columns([4, 2, 1])
        with col_id:
            st.html(
                f"<div style='font-size:0.82rem;color:#1e293b;font-family:monospace'>"
                f"{html.escape(mid[:32])}{'…' if len(mid) > 32 else ''}</div>"
                f"<div style='font-size:0.7rem;color:#94a3b8'>Joined {joined}</div>"
            )
        with col_role:
            st.html(f"<div style='margin-top:4px'>{badge}</div>")
        with col_action:
            if is_admin and mid != uid and role != "owner":
                if st.button("Remove", key=f"rm_{org.org_id}_{mid}", use_container_width=True):
                    with open_connection(cfg.db) as conn:
                        OrgRepository(conn).remove_member(org.org_id, mid)
                    st.rerun()

    st.divider()

    # ── Invite member ──────────────────────────────────────────────────────────
    if is_admin:
        st.html(
            "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
            "letter-spacing:0.06em;margin-bottom:10px'>INVITE MEMBER</div>"
        )
        with st.form("invite_member_form"):
            invite_uid = st.text_input(
                "User ID (Firebase UID)",
                placeholder="abc123xyz",
                help="The Firebase UID of the user to invite. They must have already signed up.",
            )
            invite_role = st.selectbox("Role", ["member", "admin"], index=0)
            if st.form_submit_button("Invite", type="primary"):
                iid = invite_uid.strip()
                if not iid:
                    st.error("Enter a user ID.")
                elif iid == uid:
                    st.error("You can't invite yourself.")
                else:
                    with open_connection(cfg.db) as conn:
                        OrgRepository(conn).add_member(org.org_id, iid, invite_role)
                    st.success(f"User {iid[:16]}… added as {invite_role}.")
                    st.rerun()

    # ── Rename org (owner only) ────────────────────────────────────────────────
    if is_owner:
        st.html(
            "<div style='font-size:0.72rem;font-weight:700;color:#94a3b8;"
            "letter-spacing:0.06em;margin:18px 0 10px'>SETTINGS</div>"
        )
        with st.form("rename_org_form"):
            new_name = st.text_input("Rename organization", value=org.org_name)
            if st.form_submit_button("Save"):
                name = new_name.strip()
                if name and name != org.org_name:
                    with open_connection(cfg.db) as conn:
                        OrgRepository(conn).rename_org(org.org_id, name)
                    st.success("Organization renamed.")
                    st.rerun()
