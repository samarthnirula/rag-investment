"use client";

/**
 * Lightweight gamification layer for the demo experience: confetti + toast
 * "achievement unlocked" moments. Designed to feel rewarding without being
 * heavy — short-lived toasts, no blocking modals, no network calls.
 *
 * Persists which achievements have already fired (per-browser, via
 * localStorage) so returning users don't get the same confetti twice.
 */

import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import confetti from "canvas-confetti";

export interface Achievement {
  id: string;
  title: string;
  description: string;
  emoji: string;
}

export const ACHIEVEMENTS = {
  first_question: {
    id: "first_question",
    emoji: "🎉",
    title: "First question asked",
    description: "You're off and running.",
  },
  five_questions: {
    id: "five_questions",
    emoji: "🔥",
    title: "On a roll",
    description: "5 questions in — you're getting the hang of it.",
  },
  ten_questions: {
    id: "ten_questions",
    emoji: "🏆",
    title: "Power researcher",
    description: "10 questions answered. Impressive pace.",
  },
  source_diver: {
    id: "source_diver",
    emoji: "🔍",
    title: "Source diver",
    description: "You opened your first source citation.",
  },
  explorer: {
    id: "explorer",
    emoji: "🧭",
    title: "Explorer",
    description: "You've checked out Chat, Timeline, and Overview.",
  },
} as const satisfies Record<string, Achievement>;

export type AchievementId = keyof typeof ACHIEVEMENTS;

const STORAGE_KEY = "atticus_demo_unlocked_achievements";
const CONFETTI_COLORS = ["#6366f1", "#a855f7", "#f59e0b"];

function getUnlocked(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    return new Set(JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]"));
  } catch {
    return new Set();
  }
}

function saveUnlocked(set: Set<string>) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(set)));
  } catch {
    // localStorage unavailable (private mode, etc.) — degrade silently, no persistence.
  }
}

export function useAchievements() {
  const [active, setActive] = useState<Achievement[]>([]);

  const unlock = useCallback((id: AchievementId) => {
    const achievement = ACHIEVEMENTS[id];
    if (!achievement) return;

    const unlocked = getUnlocked();
    if (unlocked.has(id)) return;
    unlocked.add(id);
    saveUnlocked(unlocked);

    setActive((prev) => [...prev, achievement]);
    confetti({
      particleCount: 90,
      spread: 75,
      startVelocity: 32,
      origin: { y: 0.25 },
      colors: CONFETTI_COLORS,
      ticks: 220,
    });

    window.setTimeout(() => {
      setActive((prev) => prev.filter((a) => a.id !== id));
    }, 4200);
  }, []);

  return { active, unlock };
}

export function AchievementToastHost({ achievements }: { achievements: Achievement[] }) {
  return (
    <div className="pointer-events-none fixed top-4 right-4 z-[80] flex flex-col gap-2 w-[min(20rem,calc(100vw-2rem))]">
      <AnimatePresence>
        {achievements.map((a) => (
          <motion.div
            key={a.id}
            initial={{ opacity: 0, x: 40, scale: 0.92 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 40, scale: 0.92 }}
            transition={{ type: "spring", stiffness: 320, damping: 26 }}
            className="pointer-events-auto flex items-center gap-3 rounded-xl border border-indigo-400/30 bg-[#13141d]/95 px-4 py-3 shadow-2xl shadow-indigo-900/40 backdrop-blur"
          >
            <div className="text-2xl leading-none">{a.emoji}</div>
            <div>
              <p className="text-sm font-semibold text-indigo-200">{a.title}</p>
              <p className="text-xs text-zinc-500">{a.description}</p>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
