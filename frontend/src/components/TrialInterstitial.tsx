"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";

export function TrialInterstitial({ onComplete }: { onComplete: () => void }) {
  const router = useRouter();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const duration = 2000;

    const frame = () => {
      const elapsed = Date.now() - start;
      const pct = Math.min(elapsed / duration, 1);
      setProgress(pct * 100);

      if (pct < 1) {
        requestAnimationFrame(frame);
      } else {
        router.push("/sign-up?plan=trial");
        onComplete();
      }
    };

    requestAnimationFrame(frame);
  }, [router, onComplete]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-[60] flex items-center justify-center bg-navy-900/95 backdrop-blur-xl"
    >
      <div className="text-center">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-gold-400 to-gold-500 mx-auto mb-4" />
        <p className="text-lg font-serif text-white mb-6">Preparing your workspace…</p>
        <div className="w-64 h-1.5 bg-white/10 rounded-full overflow-hidden mx-auto">
          <div
            className="h-full rounded-full transition-none"
            style={{
              width: `${progress}%`,
              background: "linear-gradient(90deg, #C9A84C, #B8962E)",
            }}
          />
        </div>
      </div>
    </motion.div>
  );
}
