"use client";

import { useEffect, useState } from "react";

const phrases = ["Case Research.", "Document Analysis.", "Risk Assessment."];

export function Typewriter() {
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const current = phrases[phraseIndex];

    if (!deleting && charIndex === current.length) {
      const timeout = setTimeout(() => setDeleting(true), 2000);
      return () => clearTimeout(timeout);
    }

    if (deleting && charIndex === 0) {
      setDeleting(false);
      setPhraseIndex((prev) => (prev + 1) % phrases.length);
      return;
    }

    const speed = deleting ? 40 : 80;
    const timeout = setTimeout(() => {
      setCharIndex((prev) => prev + (deleting ? -1 : 1));
    }, speed);

    return () => clearTimeout(timeout);
  }, [charIndex, deleting, phraseIndex]);

  const current = phrases[phraseIndex];
  const text = current.slice(0, charIndex);

  return (
    <span className="inline-flex items-baseline">
      <span className="bg-gradient-to-r from-gold-400 to-gold-500 bg-clip-text text-transparent">
        {text}
      </span>
      <span className="typewriter-cursor text-gold-400 ml-0.5">|</span>
    </span>
  );
}
