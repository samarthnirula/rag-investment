type SkeletonProps = {
  className?: string;
  "aria-label"?: string;
};

function cx(...classes: Array<string | undefined | false>) {
  return classes.filter(Boolean).join(" ");
}

export function Skeleton({ className, "aria-label": ariaLabel = "Loading" }: SkeletonProps) {
  return (
    <div
      aria-hidden={!ariaLabel}
      aria-label={ariaLabel}
      className={cx(
        "relative overflow-hidden rounded-md bg-white/[0.07] dark:bg-white/[0.07]",
        "before:absolute before:inset-0 before:-translate-x-full before:animate-[skeleton-shimmer_1.5s_infinite]",
        "before:bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent)]",
        className,
      )}
    />
  );
}

export function ChatSkeleton() {
  return (
    <div className="space-y-4" aria-label="Loading chat history">
      {[0, 1, 2].map((item) => (
        <div key={item} className="space-y-3">
          <div className="flex justify-end">
            <Skeleton className="h-10 w-48 rounded-xl" />
          </div>
          <div className="max-w-2xl space-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        </div>
      ))}
    </div>
  );
}
