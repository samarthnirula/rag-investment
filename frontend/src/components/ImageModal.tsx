"use client";

import { useEffect, useRef } from "react";

export interface ImageModalProps {
  src: string;
  alt: string;
  caption?: string;
  onClose: () => void;
}

export function ImageModal({ src, alt, caption, onClose }: ImageModalProps) {
  const backdropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      ref={backdropRef}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4"
      onClick={(e) => { if (e.target === backdropRef.current) onClose(); }}
    >
      <div className="relative flex max-h-full max-w-5xl flex-col overflow-hidden rounded-xl border border-white/[0.1] bg-[#0d0e17] shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between gap-4 border-b border-white/[0.08] px-4 py-3">
          <p className="truncate text-xs text-zinc-400">{caption || alt}</p>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close image"
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-zinc-500 transition-colors hover:bg-white/[0.07] hover:text-zinc-200"
          >
            ✕
          </button>
        </div>

        {/* Image */}
        <div className="flex flex-1 items-center justify-center overflow-auto bg-black/30 p-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={alt}
            className="max-h-[80vh] max-w-full rounded object-contain"
          />
        </div>
      </div>
    </div>
  );
}
