"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { demoCaseGraph, type CaseGraphData, type CaseGraphNode } from "@/lib/demo-api";

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), { ssr: false });

const GROUP_COLORS: Record<CaseGraphNode["group"], string> = {
  case: "#C9A84C",
  party: "#7FB3FF",
  issue: "#E07A5F",
  doc_type: "#81C995",
};

export function CaseGraph() {
  const [data, setData] = useState<CaseGraphData | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 480 });

  useEffect(() => {
    demoCaseGraph()
      .then(setData)
      .catch(() => setData(null));
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width } = entry.contentRect;
      setSize({ width, height: Math.max(420, Math.min(560, width * 0.55)) });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    return {
      nodes: data.nodes.map((n) => ({ ...n })),
      links: data.links.map((l) => ({ ...l })),
    };
  }, [data]);

  if (!data || data.nodes.length === 0) return null;

  return (
    <div ref={containerRef} className="w-full rounded-xl overflow-hidden border border-white/10 bg-navy-900/40">
      <ForceGraph3D
        graphData={graphData}
        width={size.width}
        height={size.height}
        backgroundColor="rgba(0,0,0,0)"
        nodeLabel="label"
        nodeColor={(n: object) => GROUP_COLORS[(n as CaseGraphNode).group] || "#999"}
        nodeVal={(n: object) => (n as CaseGraphNode).value || 1}
        linkColor={() => "rgba(255,255,255,0.15)"}
        linkOpacity={0.4}
        showNavInfo={false}
        enableNodeDrag={false}
      />
    </div>
  );
}
