"use client";

import React from "react";

// Matches **bold**, *italic*, `code`, and [Source N] citations
const INLINE_RE = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`\n]+`|\[Source\s+(\d+)\])/gi;

function renderInline(
  text: string,
  onCitationClick?: (index: number) => void,
): React.ReactNode[] {
  const parts = text.split(INLINE_RE);
  const nodes: React.ReactNode[] = [];

  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (!part) continue;

    if (part.startsWith("**") && part.endsWith("**")) {
      nodes.push(
        <strong key={i} className="font-semibold text-zinc-100">
          {part.slice(2, -2)}
        </strong>,
      );
    } else if (part.startsWith("*") && part.endsWith("*") && !part.startsWith("**")) {
      nodes.push(
        <em key={i} className="italic text-zinc-200">
          {part.slice(1, -1)}
        </em>,
      );
    } else if (part.startsWith("`") && part.endsWith("`")) {
      nodes.push(
        <code
          key={i}
          className="rounded bg-white/[0.08] px-1 py-0.5 font-mono text-[0.82em] text-amber-200"
        >
          {part.slice(1, -1)}
        </code>,
      );
    } else if (/^\[Source\s+\d+\]$/i.test(part)) {
      const n = parseInt(part.match(/\d+/)![0], 10);
      nodes.push(
        <button
          key={i}
          type="button"
          onClick={() => onCitationClick?.(n)}
          className={`mx-0.5 inline-flex h-4 min-w-[1rem] items-center justify-center rounded-full px-1 align-super text-[0.6rem] font-bold leading-none transition-colors ${
            onCitationClick
              ? "cursor-pointer bg-indigo-500/25 text-indigo-300 hover:bg-indigo-500/40"
              : "cursor-default bg-indigo-500/15 text-indigo-400"
          }`}
          title={`Jump to Source ${n}`}
        >
          {n}
        </button>,
      );
    } else {
      nodes.push(part);
    }
  }

  return nodes;
}

export function MarkdownMessage({
  content,
  onCitationClick,
}: {
  content: string;
  onCitationClick?: (index: number) => void;
}) {
  const lines = content.split("\n");
  const blocks: React.ReactNode[] = [];
  let bulletItems: string[] = [];
  let numberedItems: string[] = [];
  let codeFence: string[] | null = null;
  let codeLang = "";

  function flushBullets() {
    if (bulletItems.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="my-3 list-disc space-y-1.5 pl-5">
        {bulletItems.map((item, idx) => (
          <li key={idx}>{renderInline(item, onCitationClick)}</li>
        ))}
      </ul>,
    );
    bulletItems = [];
  }

  function flushNumbered() {
    if (numberedItems.length === 0) return;
    blocks.push(
      <ol key={`ol-${blocks.length}`} className="my-3 list-decimal space-y-1.5 pl-5">
        {numberedItems.map((item, idx) => (
          <li key={idx}>{renderInline(item, onCitationClick)}</li>
        ))}
      </ol>,
    );
    numberedItems = [];
  }

  function flushLists() {
    flushBullets();
    flushNumbered();
  }

  for (const line of lines) {
    // Fenced code block handling
    if (line.trim().startsWith("```")) {
      if (codeFence === null) {
        flushLists();
        codeLang = line.trim().slice(3).trim();
        codeFence = [];
        continue;
      } else {
        // End of fence
        const codeContent = codeFence.join("\n");
        blocks.push(
          <div key={`code-${blocks.length}`} className="my-3 overflow-x-auto rounded-lg border border-white/[0.08] bg-black/40">
            {codeLang && (
              <div className="border-b border-white/[0.06] px-3 py-1 font-mono text-[0.65rem] text-zinc-500">
                {codeLang}
              </div>
            )}
            <pre className="px-4 py-3 font-mono text-xs leading-relaxed text-zinc-300">
              <code>{codeContent}</code>
            </pre>
          </div>,
        );
        codeFence = null;
        codeLang = "";
        continue;
      }
    }

    if (codeFence !== null) {
      codeFence.push(line);
      continue;
    }

    const trimmed = line.trim();

    if (!trimmed) {
      flushLists();
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      flushNumbered();
      bulletItems.push(bullet[1]);
      continue;
    }

    const numbered = trimmed.match(/^\d+\.\s+(.+)$/);
    if (numbered) {
      flushBullets();
      numberedItems.push(numbered[1]);
      continue;
    }

    flushLists();

    // Headings
    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2];
      const className =
        level === 1
          ? "mt-4 mb-2 text-base font-bold text-white"
          : level === 2
            ? "mt-4 mb-2 text-sm font-bold text-white"
            : "mt-3 mb-1.5 text-sm font-semibold text-zinc-100";
      blocks.push(
        <div key={`h-${blocks.length}`} className={className}>
          {renderInline(text, onCitationClick)}
        </div>,
      );
      continue;
    }

    // Horizontal rule
    if (/^---+$/.test(trimmed)) {
      blocks.push(<hr key={`hr-${blocks.length}`} className="my-3 border-white/[0.08]" />);
      continue;
    }

    blocks.push(
      <p key={`p-${blocks.length}`} className="my-2">
        {renderInline(trimmed, onCitationClick)}
      </p>,
    );
  }

  flushLists();
  // Close any unclosed code fence
  if (codeFence !== null && codeFence.length > 0) {
    blocks.push(
      <pre key={`code-${blocks.length}`} className="my-3 overflow-x-auto rounded-lg border border-white/[0.08] bg-black/40 px-4 py-3 font-mono text-xs leading-relaxed text-zinc-300">
        <code>{codeFence.join("\n")}</code>
      </pre>,
    );
  }

  return <div className="text-sm text-zinc-300 leading-relaxed">{blocks}</div>;
}
