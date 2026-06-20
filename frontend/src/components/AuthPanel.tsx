"use client";

import { useState, FormEvent, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithGoogle,
  auth,
} from "@/lib/firebase";
import type { FirebaseError } from "firebase/app";

interface AuthPanelProps {
  tab: "signin" | "signup";
  onTabChange: (tab: "signin" | "signup") => void;
  onClose: () => void;
  isTrial?: boolean;
}

function isFirebaseError(err: unknown): err is FirebaseError {
  return typeof err === "object" && err !== null && "code" in err;
}

function friendlyError(err: unknown): string | null {
  const code = isFirebaseError(err) ? err.code : "";
  switch (code) {
    case "auth/invalid-credential":
    case "auth/wrong-password":
    case "auth/user-not-found":
      return "Incorrect email or password.";
    case "auth/email-already-in-use":
      return "An account with this email already exists. Try signing in instead.";
    case "auth/weak-password":
      return "Password must be at least 6 characters.";
    case "auth/invalid-email":
      return "Please enter a valid email address.";
    case "auth/too-many-requests":
      return "Too many failed attempts. Please wait a few minutes and try again.";
    case "auth/network-request-failed":
      return "Network error. Check your connection and try again.";
    case "auth/popup-blocked":
      return "Popup was blocked. Allow popups for this site in your browser settings.";
    case "auth/popup-closed-by-user":
    case "auth/cancelled-popup-request":
      return null;
    case "auth/account-exists-with-different-credential":
      return "An account with this email already exists using a different sign-in method.";
    default:
      return "Sign-in failed. Please try again.";
  }
}

function getPasswordStrength(pw: string): { score: number; label: string; color: string } {
  let score = 0;
  if (pw.length >= 6) score++;
  if (pw.length >= 10) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[0-9]/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;

  if (score <= 1) return { score: 1, label: "Weak", color: "bg-red-500" };
  if (score <= 2) return { score: 2, label: "Fair", color: "bg-amber-500" };
  if (score <= 3) return { score: 3, label: "Good", color: "bg-yellow-400" };
  if (score <= 4) return { score: 4, label: "Strong", color: "bg-emerald-400" };
  return { score: 5, label: "Excellent", color: "bg-emerald-400" };
}

function FloatingInput({
  label,
  type = "text",
  value,
  onChange,
  autoComplete,
  required,
  minLength,
  showToggle,
}: {
  label: string;
  type?: string;
  value: string;
  onChange: (v: string) => void;
  autoComplete?: string;
  required?: boolean;
  minLength?: number;
  showToggle?: boolean;
}) {
  const [focused, setFocused] = useState(false);
  const [visible, setVisible] = useState(false);
  const active = focused || value.length > 0;
  const inputType = showToggle && visible ? "text" : type;

  return (
    <div className="relative">
      <input
        type={inputType}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        autoComplete={autoComplete}
        required={required}
        minLength={minLength}
        className="peer w-full bg-white/[0.05] border border-white/[0.09] rounded-lg px-3 pt-5 pb-2 text-sm text-zinc-200 focus:outline-none focus:border-gold-400/50 focus:bg-white/[0.07] transition-colors"
      />
      <label
        className={`absolute left-3 transition-all duration-200 pointer-events-none ${
          active
            ? "top-1.5 text-[0.6rem] text-gold-400"
            : "top-3.5 text-sm text-zinc-600"
        }`}
      >
        {label}
      </label>
      {showToggle && value.length > 0 && (
        <button
          type="button"
          onClick={() => setVisible((v) => !v)}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          tabIndex={-1}
        >
          {visible ? "Hide" : "Show"}
        </button>
      )}
    </div>
  );
}

export function AuthPanel({ tab, onTabChange, onClose, isTrial = false }: AuthPanelProps) {
  const router = useRouter();
  const formRef = useRef<HTMLDivElement>(null);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [shaking, setShaking] = useState(false);

  useEffect(() => {
    setEmail("");
    setPassword("");
    setConfirm("");
    setError(null);
  }, [tab]);

  useEffect(() => {
    if (error && tab === "signin") {
      setShaking(true);
      const t = setTimeout(() => setShaking(false), 500);
      return () => clearTimeout(t);
    }
  }, [error, tab]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    if (tab === "signup" && password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      if (!auth) throw new Error("Firebase not configured");
      if (tab === "signin") {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        await createUserWithEmailAndPassword(auth, email, password);
      }
      router.push("/chat");
    } catch (err) {
      const msg = friendlyError(err);
      if (msg) setError(msg);
    } finally {
      setLoading(false);
    }
  }

  async function handleGoogle() {
    setError(null);
    setLoading(true);
    try {
      await signInWithGoogle();
      router.push("/chat");
    } catch (err) {
      const msg = friendlyError(err);
      if (msg) setError(msg);
      setLoading(false);
    }
  }

  const strength = tab === "signup" && password.length > 0 ? getPasswordStrength(password) : null;

  return (
    <div
      className="fixed inset-0 z-50 flex"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* Left panel: animated gradient */}
      <div className="hidden md:flex w-1/2 gradient-mesh items-center justify-center relative overflow-hidden">
        <div className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: `linear-gradient(rgba(201,168,76,0.5) 1px, transparent 1px),
              linear-gradient(90deg, rgba(201,168,76,0.5) 1px, transparent 1px)`,
            backgroundSize: "48px 48px",
          }}
        />
        <div className="relative text-center px-12">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-gold-400 to-gold-500 mx-auto mb-6 shadow-[0_0_40px_rgba(201,168,76,0.3)]" />
          <h2 className="text-3xl font-serif text-white mb-3">Atticus</h2>
          <p className="text-sm text-zinc-400 leading-relaxed max-w-xs mx-auto">
            AI-powered legal research with exact page citations.
            Built for attorneys who can&apos;t afford to miss a detail.
          </p>
        </div>
      </div>

      {/* Right panel: form */}
      <div
        className="flex-1 bg-navy-900 flex items-center justify-center p-8"
        onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      >
        <div
          ref={formRef}
          className={`w-full max-w-sm ${shaking ? "animate-shake" : ""}`}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2 md:hidden">
              <div className="w-4 h-4 rounded-sm bg-gradient-to-br from-gold-400 to-gold-500" />
              <span className="text-sm font-semibold text-zinc-200">Atticus</span>
            </div>
            <button
              onClick={onClose}
              className="text-zinc-500 hover:text-zinc-300 transition-colors leading-none p-1 ml-auto"
              aria-label="Close"
            >
              <svg width="20" height="20" viewBox="0 0 16 16" fill="none">
                <path d="M12 4L4 12M4 4l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </button>
          </div>

          {isTrial && tab === "signup" && (
            <div className="mb-4">
              <div className="inline-flex items-center gap-1.5 text-[0.65rem] font-bold text-gold-400 border border-gold-500/30 bg-gold-500/10 rounded-full px-2.5 py-1 tracking-wide mb-2">
                FREE TRIAL
              </div>
              <h2 className="text-xl font-serif text-white leading-tight">
                Start your free trial
              </h2>
              <p className="text-xs text-zinc-500 mt-0.5">No credit card required.</p>
            </div>
          )}

          {/* Tabs */}
          <div className="flex border-b border-white/[0.07] mb-6">
            {(["signin", "signup"] as const).map((t) => (
              <button
                key={t}
                onClick={() => onTabChange(t)}
                className={`flex-1 pb-3 text-xs font-semibold tracking-wide transition-colors relative ${
                  tab === t ? "text-gold-400" : "text-zinc-600 hover:text-zinc-400"
                }`}
              >
                {t === "signin" ? "Sign In" : "Create Account"}
                {tab === t && (
                  <div className="absolute bottom-0 left-0 right-0 h-px bg-gold-400" />
                )}
              </button>
            ))}
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-3">
            <FloatingInput
              label="Email"
              type="email"
              value={email}
              onChange={setEmail}
              autoComplete="email"
              required
            />

            <FloatingInput
              label={tab === "signup" ? "Password (min. 6 characters)" : "Password"}
              type="password"
              value={password}
              onChange={setPassword}
              autoComplete={tab === "signup" ? "new-password" : "current-password"}
              required
              minLength={tab === "signup" ? 6 : undefined}
              showToggle
            />

            {/* Password strength bar */}
            {strength && (
              <div className="space-y-1">
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((level) => (
                    <div
                      key={level}
                      className={`h-1 flex-1 rounded-full transition-colors ${
                        level <= strength.score ? strength.color : "bg-white/[0.08]"
                      }`}
                    />
                  ))}
                </div>
                <p className="text-[0.6rem] text-zinc-600">{strength.label}</p>
              </div>
            )}

            {tab === "signup" && (
              <FloatingInput
                label="Confirm password"
                type="password"
                value={confirmPassword}
                onChange={setConfirm}
                autoComplete="new-password"
                required
              />
            )}

            {error && (
              <div className="flex items-start gap-1.5 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                <span className="w-1 h-1 rounded-full bg-red-400 shrink-0 mt-1.5" />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-gold-400 to-gold-500 text-navy-900 font-semibold text-sm rounded-lg px-4 py-2.5 mt-1 transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-60 cursor-pointer"
            >
              {loading
                ? "Please wait…"
                : tab === "signin"
                ? "Sign In"
                : isTrial
                ? "Start free trial"
                : "Create Account"}
            </button>

            <div className="flex items-center gap-3 my-1">
              <div className="flex-1 h-px bg-white/[0.07]" />
              <span className="text-xs text-zinc-700">or</span>
              <div className="flex-1 h-px bg-white/[0.07]" />
            </div>

            <button
              type="button"
              onClick={handleGoogle}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2.5 bg-white/[0.06] hover:bg-white/[0.10] disabled:opacity-50 border border-white/[0.10] hover:border-white/[0.16] rounded-lg px-4 py-2.5 text-sm text-zinc-300 transition-colors cursor-pointer"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
              </svg>
              Continue with Google
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
