import Link from "next/link";

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-[#08090e] text-white">
      <nav className="flex items-center justify-between px-8 py-4 border-b border-white/[0.06]">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-5 h-5 rounded bg-gradient-to-br from-indigo-500 to-purple-500" />
          <span className="text-sm font-bold text-zinc-100 tracking-tight">Atticus</span>
        </Link>
      </nav>

      <main className="max-w-2xl mx-auto px-8 py-16">
        <h1 className="text-4xl font-extrabold text-white mb-2 tracking-tight">Terms of Service</h1>
        <p className="text-xs text-zinc-500 mb-10">Version 1 · Effective June 2026</p>

        <div className="space-y-8 text-sm text-zinc-400 leading-relaxed">
          <section>
            <h2 className="text-sm font-bold text-white mb-2">1. Acceptance</h2>
            <p>
              By creating an account or using Atticus, you agree to these Terms. If you do not agree,
              do not use the service.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">2. Description of service</h2>
            <p>
              Atticus is an AI-powered legal research tool. It ingests documents you upload and
              provides AI-generated answers with citations. Outputs are research aids only and do not
              constitute legal advice.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">3. No legal advice</h2>
            <p>
              Nothing produced by Atticus constitutes legal advice or creates an attorney-client
              relationship. Always consult a licensed attorney for legal decisions.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">4. Your content</h2>
            <p>
              You retain all rights to documents you upload. By uploading, you grant Atticus a limited
              license to process and store your content solely to provide the service. We do not use
              your documents to train AI models.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">5. Acceptable use</h2>
            <p>You may not use Atticus to process illegal content, infringe third-party rights,
            circumvent security controls, or engage in any activity that violates applicable law.</p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">6. Billing and plans</h2>
            <p>
              Paid plans are billed monthly. Unused queries do not roll over. We may change pricing
              with 30 days' notice.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">7. Limitation of liability</h2>
            <p>
              To the maximum extent permitted by law, Atticus is not liable for indirect, incidental,
              or consequential damages. Our total liability is capped at amounts paid in the preceding
              3 months.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">8. Termination</h2>
            <p>
              We may suspend or terminate your account for material breach of these Terms. You may
              cancel at any time. On termination, your data is deleted within 30 days on request.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">9. Governing law</h2>
            <p>These Terms are governed by the laws of Delaware, USA.</p>
          </section>
        </div>

        <div className="mt-12 pt-8 border-t border-white/[0.06]">
          <Link href="/" className="text-sm text-indigo-300 hover:text-indigo-200 transition-colors">
            ← Back to home
          </Link>
        </div>
      </main>
    </div>
  );
}
