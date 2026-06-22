"use client";

import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import { AuthPanel } from "@/components/AuthPanel";
import { Typewriter } from "@/components/Typewriter";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const ScalesOfJustice = dynamic(
  () => import("@/components/ScalesOfJustice").then((m) => m.ScalesOfJustice),
  { ssr: false }
);

const CaseGraph = dynamic(
  () => import("@/components/CaseGraph").then((m) => m.CaseGraph),
  { ssr: false }
);

function LandingPageInner() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const searchParams = useSearchParams();

  const [showAuth, setShowAuth] = useState(false);
  const [tab, setTab] = useState<"signin" | "signup">("signin");
  const [isTrial, setIsTrial] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    if (!loading && user) router.push("/chat");
  }, [user, loading, router]);

  useEffect(() => {
    if (searchParams.get("plan") === "trial") {
      setTab("signup");
      setIsTrial(true);
      setShowAuth(true);
    }
  }, [searchParams]);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  function handleTrialSignup() {
    router.push("/demo");
  }

  function handleSignIn() {
    setIsTrial(false);
    setTab("signin");
    setShowAuth(true);
  }

  if (loading) return null;

  return (
    <div className="min-h-screen bg-navy-900 text-white font-sans">
      {/* ── STICKY NAV ──────────────────────────────────────── */}
      <nav
        className={`fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-8 py-4 transition-all duration-300 ${
          scrolled
            ? "backdrop-blur-lg bg-navy-900/80 border-b border-white/[0.06] shadow-lg"
            : "border-b border-transparent"
        }`}
      >
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded bg-gradient-to-br from-gold-400 to-gold-500 shadow-[0_0_12px_rgba(201,168,76,0.6)]" />
          <span className="text-sm font-bold text-zinc-100 tracking-tight">Atticus</span>
          <span className="text-xs text-gold-400 ml-1">Legal Research Intelligence</span>
        </div>
        <div className="flex items-center gap-2">
          <a href="/about" className="text-xs text-zinc-400 hover:text-zinc-200 px-3 py-2 transition-colors">About</a>
          <a href="/terms" className="text-xs text-zinc-400 hover:text-zinc-200 px-3 py-2 transition-colors">Terms</a>
          <a href="/privacy" className="text-xs text-zinc-400 hover:text-zinc-200 px-3 py-2 transition-colors">Privacy</a>
          <button
            onClick={handleSignIn}
            className="text-xs font-medium text-zinc-300 border border-white/[0.15] rounded-lg px-4 py-2 hover:bg-white/[0.07] transition-colors"
          >
            Sign in
          </button>
        </div>
      </nav>

      {/* ── AUTH PANEL ─────────────────────────────────────── */}
      {showAuth && (
        <AuthPanel
          tab={tab}
          onTabChange={(t) => { setTab(t); if (t === "signin") setIsTrial(false); }}
          onClose={() => { setShowAuth(false); setIsTrial(false); }}
          isTrial={isTrial}
        />
      )}

      {/* ── HERO ────────────────────────────────────────────── */}
      <section className="relative overflow-hidden gradient-mesh pt-20">
        {/* Grid texture */}
        <div className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: `linear-gradient(rgba(201,168,76,0.5) 1px, transparent 1px),
              linear-gradient(90deg, rgba(201,168,76,0.5) 1px, transparent 1px)`,
            backgroundSize: "48px 48px",
          }}
        />
        {/* Glow blobs */}
        <div className="absolute -top-32 right-[10%] w-[500px] h-[500px] rounded-full bg-[radial-gradient(circle,rgba(201,168,76,0.08),transparent_70%)]" />
        <div className="absolute -bottom-20 left-[5%] w-[400px] h-[400px] rounded-full bg-[radial-gradient(circle,rgba(139,92,246,0.08),transparent_70%)]" />

        <div className="relative grid grid-cols-2 gap-8 max-w-6xl mx-auto px-12 py-16">
          {/* Left: headline */}
          <div className="py-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-gold-400 border border-gold-500/30 bg-gold-500/10 rounded-full px-3 py-1.5 mb-6 tracking-wide">
                ⚡ LEGAL AI · EARLY ACCESS
              </div>
              <h1 className="font-serif text-[72px] leading-[1.05] tracking-tight mb-3">
                <span className="bg-gradient-to-r from-white to-gold-400 bg-clip-text text-transparent">
                  Find the facts.
                </span>
              </h1>
              <div className="text-[72px] font-serif leading-[1.05] tracking-tight mb-5">
                <Typewriter />
              </div>
              <p className="text-base text-zinc-400 leading-relaxed max-w-md mb-8">
                Atticus reads your case files, flags risk clauses, and answers legal questions
                with exact page citations. Built for attorneys who can&apos;t afford to miss a detail.
              </p>
              <button
                onClick={handleTrialSignup}
                className="bg-gradient-to-r from-gold-400 to-gold-500 text-navy-900 font-semibold text-sm rounded-full px-8 py-3.5 transition-all hover:scale-[1.02] active:scale-[0.98]"
                style={{ animation: "glow-pulse 2s ease-in-out infinite" }}
              >
                Start Free Trial
              </button>
            </motion.div>
          </div>

          {/* Right: Three.js scales */}
          <motion.div
            className="py-8 px-4 h-[420px]"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <ScalesOfJustice />
          </motion.div>
        </div>
      </section>

      {/* ── STATS BAR ──────────────────────────────────────── */}
      <div className="bg-navy-800 border-t border-b border-white/[0.07] px-12 py-7">
        <div className="flex justify-around flex-wrap gap-5 max-w-4xl mx-auto">
          {[
            ["Free trial", "No credit card required", "text-gold-400"],
            ["100%", "Your docs, your answers", "text-emerald-300"],
            ["8-stage", "Hybrid retrieval pipeline", "text-gold-400"],
            ["GDPR", "Data deleted on request", "text-emerald-400"],
          ].map(([val, label, color]) => (
            <div key={label} className="text-center">
              <div className={`text-3xl font-extrabold tracking-tight ${color}`}>{val}</div>
              <div className="text-xs text-zinc-600 mt-1">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── POSITIONING ─────────────────────────────────────── */}
      <section className="bg-navy-900 px-12 py-16">
        <div className="max-w-3xl mx-auto text-center mb-10">
          <p className="text-xs font-bold text-gold-500 tracking-[.1em] mb-3">BUILT FOR SMALL LEGAL TEAMS</p>
          <h2 className="text-3xl font-extrabold text-white tracking-tight mb-4 leading-tight">
            Research-grade AI without the enterprise wall.
          </h2>
          <p className="text-sm text-zinc-500 leading-relaxed max-w-xl mx-auto">
            Atticus keeps the pieces attorneys actually need: cited answers, organized matters, document review,
            timeline extraction, contradiction spotting, and a free trial before anyone books a call.
          </p>
        </div>
        <div className="grid grid-cols-4 gap-4 max-w-4xl mx-auto mb-5">
          {[
            ["Free trial first", "Test the system before subscribing. No credit card, no sales call."],
            ["Matter workspace", "Keep cases, uploads, chats, notes, and research artifacts together."],
            ["Contradictions + timelines", "Surface conflicting facts and convert filings into case chronology."],
            ["Cost guardrails", "Usage limits and cost telemetry protect a 60% margin target as upload volume grows."],
          ].map(([title, body]) => (
            <div key={title} className="backdrop-blur-md bg-white/5 border border-white/10 rounded-lg p-4 text-left">
              <div className="text-sm font-bold text-white mb-2">{title}</div>
              <div className="text-xs text-zinc-500 leading-relaxed">{body}</div>
            </div>
          ))}
        </div>
        <div className="flex justify-center gap-2 flex-wrap">
          {["Cited answers","Hybrid search","PDF + OCR","Client-ready summaries","Audit-aware activity"].map((t) => (
            <span key={t} className="text-xs text-gold-400 bg-gold-500/10 border border-gold-500/20 rounded-full px-3 py-1.5">{t}</span>
          ))}
        </div>
      </section>

      {/* ── FEATURES ───────────────────────────────────────── */}
      <section className="bg-navy-900 px-12 py-20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <p className="text-xs font-bold text-gold-500 tracking-[.1em] mb-3">WHAT YOU GET</p>
            <h2 className="text-3xl font-extrabold text-white tracking-tight leading-tight">Everything a litigator needs.</h2>
          </div>
          <div className="grid grid-cols-4 gap-4">
            {[
              ["⚖️", "Case Chat", "Ask questions in plain English. Get cited answers from the exact pages of your documents."],
              ["🚩", "Risk Flags", "Automatic risk scoring for contracts. High, medium, low ratings with clause and page references."],
              ["📅", "Case Timeline", "Auto-structured visual timelines from your case files. Filter by event type instantly."],
              ["💬", "Team Forum", "Per-case discussion board for your team. Post findings, export as a markdown report."],
            ].map(([icon, title, body]) => (
              <div key={title} className="backdrop-blur-md bg-white/5 border border-white/10 rounded-xl p-5">
                <div className="text-2xl mb-3">{icon}</div>
                <div className="text-sm font-semibold text-zinc-100 mb-2">{title}</div>
                <div className="text-xs text-zinc-500 leading-relaxed">{body}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CASE GRAPH ──────────────────────────────────────── */}
      <section className="bg-navy-900 px-12 py-20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <p className="text-xs font-bold text-gold-500 tracking-[.1em] mb-3">SEE IT IN ACTION</p>
            <h2 className="text-3xl font-extrabold text-white tracking-tight leading-tight">
              How Atticus maps a case.
            </h2>
            <p className="text-xs text-zinc-500 mt-3 max-w-xl mx-auto">
              A live view of the public-record demo case structure — parties, key issues, and document
              categories. No user data, queries, or private documents are ever shown here.
            </p>
          </div>
          <CaseGraph />
        </div>
      </section>

      {/* ── HOW IT WORKS ────────────────────────────────────── */}
      <section className="bg-navy-800 border-t border-b border-white/[0.06] px-12 py-20">
        <div className="max-w-3xl mx-auto text-center mb-14">
          <p className="text-xs font-bold text-gold-500 tracking-[.1em] mb-3">HOW IT WORKS</p>
          <h2 className="text-3xl font-extrabold text-white tracking-tight leading-tight">Three steps. Seconds to your first answer.</h2>
        </div>
        <div className="grid grid-cols-3 gap-10 max-w-3xl mx-auto">
          {[
            ["1", "Upload your files", "PDF, scanned or digital. OCR handles everything. Text, tables, images, footnotes."],
            ["2", "Ask any question", "Plain English. No query language. 'What did the defendant say about wire transfers?'"],
            ["3", "Get cited answers", "Every answer cites the exact source page. Click to expand. Export as markdown."],
          ].map(([n, t, d]) => (
            <div key={n} className="text-center">
              <div className="w-11 h-11 rounded-lg bg-gold-500/15 border border-gold-500/30 text-gold-400 font-bold text-lg flex items-center justify-center mx-auto mb-4">{n}</div>
              <div className="text-sm font-semibold text-zinc-100 mb-2">{t}</div>
              <div className="text-xs text-zinc-600 leading-relaxed">{d}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── PRICING ─────────────────────────────────────────── */}
      <section className="bg-navy-900 px-12 py-20">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <p className="text-xs font-bold text-gold-500 tracking-[.1em] mb-3">PRICING</p>
            <h2 className="text-3xl font-extrabold text-white tracking-tight leading-tight">Simple. No surprises.</h2>
          </div>
          <div className="grid grid-cols-3 gap-5 max-w-4xl mx-auto">
            {[
              {
                tier: "SOLO", price: "$79", sub: "/month", popular: false,
                features: ["Unlimited uploads","500 AI queries/mo","Case Board & Timeline","PDF + OCR"],
                bg: "backdrop-blur-md bg-white/5", border: "border-white/10", tc: "text-zinc-500", vc: "text-white",
              },
              {
                tier: "FIRM", price: "$149", sub: "/seat/month", popular: true,
                features: ["Everything in Solo","Shared Case Boards","Team Discussion","Priority support","Audit export"],
                bg: "bg-gold-500/10", border: "border-gold-500/40", tc: "text-gold-400", vc: "text-white",
              },
              {
                tier: "ENTERPRISE", price: "Custom", sub: "", popular: false,
                features: ["Everything in Firm","SSO / SAML","Custom retention","Onboarding","SLA & DPA"],
                bg: "backdrop-blur-md bg-white/5", border: "border-white/10", tc: "text-zinc-500", vc: "text-white",
              },
            ].map(({ tier, price, sub, popular, features, bg, border, tc, vc }) => (
              <div key={tier} className={`${bg} ${border} border rounded-xl p-6 relative`}>
                {popular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-gradient-to-r from-gold-400 to-gold-500 text-navy-900 text-[0.6rem] font-bold px-3 py-1 rounded-full whitespace-nowrap">MOST POPULAR</div>
                )}
                <p className={`text-xs font-bold ${tc} tracking-wide mb-3`}>{tier}</p>
                <div className={`text-3xl font-extrabold ${vc} mb-1`}>{price}</div>
                <div className={`text-xs ${tc} mb-5`}>{sub}</div>
                <ul className="space-y-2">
                  {features.map((f) => (
                    <li key={f} className={`text-xs ${tc} flex items-center gap-1.5`}>
                      <span className="text-emerald-400">✓</span> {f}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── FINAL CTA ──────────────────────────────────────── */}
      <section className="bg-navy-800 border-t border-white/[0.06] px-12 py-20 text-center">
        <h2 className="text-4xl font-extrabold text-white tracking-tight mb-4 leading-tight">
          Every case deserves this level of research.
        </h2>
        <p className="text-sm text-zinc-600 leading-relaxed mb-10">
          Start your free trial. No credit card required.
        </p>
        <button
          onClick={handleTrialSignup}
          className="bg-gradient-to-r from-gold-400 to-gold-500 text-navy-900 font-semibold text-sm rounded-full px-8 py-3.5 transition-all hover:scale-[1.02] active:scale-[0.98]"
          style={{ animation: "glow-pulse 2s ease-in-out infinite" }}
        >
          Start Free Trial
        </button>
      </section>

      {/* ── FOOTER ──────────────────────────────────────────── */}
      <footer className="bg-navy-900 border-t border-white/[0.06] px-12 py-6 flex items-center justify-between flex-wrap gap-3">
        <span className="text-xs text-zinc-700">
          <strong className="text-zinc-500">Atticus</strong> © 2026 Atticus
        </span>
        <span className="text-xs text-zinc-700">AI responses are research aids only, not legal advice.</span>
        <div className="flex gap-5">
          {["Terms", "Privacy", "Security"].map((l) => (
            <a key={l} href={`/${l.toLowerCase()}`} className="text-xs text-zinc-700 hover:text-zinc-500 transition-colors">{l}</a>
          ))}
        </div>
      </footer>
    </div>
  );
}

export default function LandingPage() {
  return (
    <Suspense>
      <LandingPageInner />
    </Suspense>
  );
}
