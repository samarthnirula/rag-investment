"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { getAnalytics, AnalyticsData } from "@/lib/api";
import { useMotionValue, useTransform, motion, animate } from "framer-motion";

function Spinner() {
  return (
    <div className="flex gap-1.5">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

function AnimatedNumber({ value, prefix = "", suffix = "", decimals = 0 }: {
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
}) {
  const motionVal = useMotionValue(0);
  const rounded = useTransform(motionVal, (v) =>
    decimals > 0 ? v.toFixed(decimals) : Math.round(v).toLocaleString()
  );
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const controls = animate(motionVal, value, {
      duration: 1.2,
      ease: "easeOut",
    });
    return () => controls.stop();
  }, [value, motionVal]);

  return (
    <span ref={ref}>
      {prefix}
      <motion.span>{rounded}</motion.span>
      {suffix}
    </span>
  );
}

function KpiCard({
  label,
  value,
  numericValue,
  accent,
  prefix,
  suffix,
  decimals,
}: {
  label: string;
  value: string;
  numericValue?: number;
  accent?: boolean;
  prefix?: string;
  suffix?: string;
  decimals?: number;
}) {
  return (
    <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-5">
      <p className="text-[0.65rem] font-bold text-zinc-500 tracking-widest mb-2 uppercase">
        {label}
      </p>
      <p className={`text-2xl font-bold ${accent ? "text-indigo-300" : "text-white"}`}>
        {numericValue !== undefined ? (
          <AnimatedNumber value={numericValue} prefix={prefix} suffix={suffix} decimals={decimals} />
        ) : (
          value
        )}
      </p>
    </div>
  );
}

function MiniBarChart({ data, labelKey, valueKey }: {
  data: Array<Record<string, string | number>>;
  labelKey: string;
  valueKey: string;
}) {
  if (!data.length) return <p className="text-xs text-zinc-600">No data.</p>;
  const max = Math.max(...data.map((d) => Number(d[valueKey])), 1);
  return (
    <div className="space-y-1.5">
      {data.map((d, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="text-[0.65rem] text-zinc-500 w-20 truncate shrink-0">
            {String(d[labelKey])}
          </span>
          <div className="flex-1 h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all"
              style={{ width: `${(Number(d[valueKey]) / max) * 100}%` }}
            />
          </div>
          <span className="text-[0.65rem] text-zinc-500 w-8 text-right shrink-0">
            {d[valueKey]}
          </span>
        </div>
      ))}
    </div>
  );
}

function SparkLine({ data }: { data: Array<{ date: string; Queries: number }> }) {
  if (!data.length) return <p className="text-xs text-zinc-600">No queries in this period.</p>;
  const max = Math.max(...data.map((d) => d.Queries), 1);
  const w = 600;
  const h = 80;
  const pts = data.map((d, i) => {
    const x = (i / (data.length - 1 || 1)) * w;
    const y = h - (d.Queries / max) * (h - 8) - 4;
    return `${x},${y}`;
  });
  const gradientId = "sparkline-gradient";
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16" preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(99,102,241)" stopOpacity="0.3" />
          <stop offset="100%" stopColor="rgb(99,102,241)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        fill={`url(#${gradientId})`}
        stroke="none"
        points={`0,${h} ${pts.join(" ")} ${w},${h}`}
      />
      <polyline
        fill="none"
        stroke="rgb(99 102 241)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={pts.join(" ")}
      />
    </svg>
  );
}

const RANGE_OPTIONS: Record<string, number> = {
  "Last 7 days": 7,
  "Last 30 days": 30,
  "Last 60 days": 60,
  "Last 90 days": 90,
};

export default function DataPage() {
  const { user, idToken, loading } = useAuth();
  const router = useRouter();
  const [rangeLabel, setRangeLabel] = useState("Last 30 days");
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loadingData, setLoadingData] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/");
  }, [user, loading, router]);

  const loadAnalytics = useCallback(async () => {
    if (!idToken) return;
    setLoadingData(true);
    setError(null);
    try {
      const days = RANGE_OPTIONS[rangeLabel] ?? 30;
      const data = await getAnalytics(idToken, days);
      setAnalytics(data);
    } catch {
      setError("Could not load analytics. Make sure the backend is running.");
    } finally {
      setLoadingData(false);
    }
  }, [idToken, rangeLabel]);

  useEffect(() => {
    if (idToken) loadAnalytics();
  }, [idToken, loadAnalytics]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner />
      </div>
    );
  }

  if (!user) return null;

  const s = analytics?.stats;
  const MARGIN_CAP = 15;
  const periodQueries = (analytics?.daily || []).reduce((a, d) => a + d.Queries, 0);
  const totalCost = (s?.estimated_cost_usd ?? 0) + (analytics?.upload_cost_this_month ?? 0);

  return (
    <div className="px-8 py-8 overflow-y-auto">
      <div className="max-w-4xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-xl font-bold text-white">Data Usage</h1>
            <p className="text-sm text-zinc-500 mt-0.5">Query analytics and usage metrics.</p>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={rangeLabel}
              onChange={(e) => setRangeLabel(e.target.value)}
              className="text-xs bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-1.5 text-zinc-400 focus:outline-none focus:border-indigo-500/40"
            >
              {Object.keys(RANGE_OPTIONS).map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
            <button
              onClick={loadAnalytics}
              disabled={loadingData}
              className="text-xs text-zinc-500 hover:text-zinc-300 border border-white/[0.08] rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
            >
              ↺ Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-3 text-xs text-red-300 mb-6">
            {error}
          </div>
        )}

        {loadingData && !analytics && (
          <div className="flex py-16 justify-center">
            <Spinner />
          </div>
        )}

        {analytics && (
          <>
            {/* KPI grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              <KpiCard label="Total queries" value="" numericValue={s?.total_queries ?? 0} accent />
              <KpiCard label={`Queries (${rangeLabel.toLowerCase()})`} value="" numericValue={periodQueries} />
              <KpiCard label="Queries today" value="" numericValue={s?.queries_today ?? 0} />
              <KpiCard label="Queries this month" value="" numericValue={s?.queries_this_month ?? 0} />
            </div>

            <div className="grid grid-cols-2 gap-3 mb-6">
              <KpiCard
                label="Estimated AI cost"
                value=""
                numericValue={totalCost}
                prefix="$"
                decimals={2}
                accent
              />
              <KpiCard
                label="60% margin cost cap"
                value={`$${MARGIN_CAP.toFixed(2)}`}
              />
            </div>

            <p className="text-[0.65rem] text-zinc-600 mb-6">
              Uploads this month: {analytics.uploads_this_month} · estimated upload cost: ${analytics.upload_cost_this_month.toFixed(2)}.
              These are planning estimates, not provider invoices.
            </p>

            {/* Queries over time */}
            <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5 mb-4">
              <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3 uppercase">
                Queries over time — {rangeLabel}
              </p>
              <SparkLine data={analytics.daily ?? []} />
              {analytics.daily?.length > 0 && (
                <div className="flex justify-between mt-1">
                  <span className="text-[0.6rem] text-zinc-700">{analytics.daily[0]?.date}</span>
                  <span className="text-[0.6rem] text-zinc-700">{analytics.daily[analytics.daily.length - 1]?.date}</span>
                </div>
              )}
            </div>

            {/* Breakdowns */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5">
                <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3 uppercase">
                  By chat type
                </p>
                <MiniBarChart
                  data={analytics.by_page ?? []}
                  labelKey="Page"
                  valueKey="Queries"
                />
              </div>
              <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5">
                <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3 uppercase">
                  Chunks retrieved (total)
                </p>
                <p className="text-2xl font-bold text-emerald-300">
                  <AnimatedNumber value={s?.chunks_retrieved ?? 0} />
                </p>
                <p className="text-xs text-zinc-600 mt-1">Across all queries</p>
              </div>
            </div>

            {/* Recent queries */}
            <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5">
              <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3 uppercase">
                Recent queries
              </p>
              {(analytics.recent ?? []).length === 0 && (
                <p className="text-xs text-zinc-600">No queries yet. Start chatting to see data here.</p>
              )}
              <div className="space-y-2">
                {(analytics.recent ?? []).slice(0, 15).map((r, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 py-2.5 border-b border-white/[0.04] last:border-0"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-zinc-300 truncate">{r.Query}</p>
                      <p className="text-[0.65rem] text-zinc-600 mt-0.5">
                        {r.Time} · {r.Sources} sources · {r["Resp len"]} chars
                        {r.Chat && <span className="ml-1">· {r.Chat}</span>}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
