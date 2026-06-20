import Link from "next/link";

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-[#08090e] text-white">
      <nav className="flex items-center justify-between px-8 py-4 border-b border-white/[0.06]">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-5 h-5 rounded bg-gradient-to-br from-indigo-500 to-purple-500" />
          <span className="text-sm font-bold text-zinc-100 tracking-tight">Atticus</span>
        </Link>
      </nav>

      <main className="max-w-2xl mx-auto px-8 py-16">
        <h1 className="text-4xl font-extrabold text-white mb-2 tracking-tight">Privacy Policy</h1>
        <p className="text-xs text-zinc-500 mb-10">Version 1 · Effective June 2026</p>

        <div className="space-y-8 text-sm text-zinc-400 leading-relaxed">
          <section>
            <h2 className="text-sm font-bold text-white mb-2">1. What we collect</h2>
            <ul className="space-y-1.5 list-disc list-inside text-zinc-500">
              <li>Account information: email address, display name, Firebase UID</li>
              <li>Documents you upload for processing</li>
              <li>Query logs: text of queries, response lengths, source counts, timestamps</li>
              <li>Usage telemetry: query counts, upload counts, estimated AI costs</li>
            </ul>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">2. How we use your data</h2>
            <ul className="space-y-1.5 list-disc list-inside text-zinc-500">
              <li>To provide the Atticus service — answer queries, manage cases, store chats</li>
              <li>To monitor usage against plan limits</li>
              <li>To detect abuse and enforce our Terms of Service</li>
              <li>To improve the service (aggregate, anonymized usage only)</li>
            </ul>
            <p className="mt-3">
              We do not use your uploaded documents or query content to train AI models.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">3. Data sharing</h2>
            <p>
              We do not sell your data. We share data only with:
            </p>
            <ul className="mt-2 space-y-1.5 list-disc list-inside text-zinc-500">
              <li>Anthropic — your queries are sent to Claude to generate answers</li>
              <li>Firebase / Google — for authentication</li>
              <li>Our cloud infrastructure provider — for hosting and storage</li>
            </ul>
            <p className="mt-3">
              All sub-processors are bound by data processing agreements and GDPR-compliant transfer
              mechanisms.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">4. Your rights (GDPR)</h2>
            <ul className="space-y-1.5 list-disc list-inside text-zinc-500">
              <li>Access: request a copy of your data at any time</li>
              <li>Rectification: correct inaccurate personal data</li>
              <li>Erasure (Art. 17): delete all your data from your Profile page</li>
              <li>Portability: export your query history as Markdown</li>
              <li>Object: object to processing for direct marketing (we don't do this)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">5. Data retention</h2>
            <p>
              Query logs and chat history are retained while your account is active. Documents are
              retained until you delete them or close your account. On account deletion, all personal
              data is deleted within 30 days.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">6. Security</h2>
            <p>
              We use industry-standard encryption in transit (TLS 1.2+) and at rest. Access to
              production systems is restricted to authorized personnel with MFA enabled.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">7. Cookies</h2>
            <p>
              We use essential session cookies only. No advertising or tracking cookies.
            </p>
          </section>

          <section>
            <h2 className="text-sm font-bold text-white mb-2">8. Contact</h2>
            <p>
              For privacy questions or data requests, contact us at the email address registered on
              your account.
            </p>
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
