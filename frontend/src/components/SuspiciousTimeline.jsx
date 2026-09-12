import { AlertTriangle } from "lucide-react";

/**
 * Renders detected "suspicious" (high-attention) regions as a horizontal
 * timeline bar with highlighted segments, plus a scannable list below.
 */
export default function SuspiciousTimeline({ regions = [], durationSec = 1 }) {
  if (!regions.length) {
    return (
      <div className="glass-card p-4 text-sm text-slate-400">
        No strongly suspicious regions were flagged for this clip.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="relative h-3 rounded-full bg-white/5 overflow-hidden border border-white/10">
        {regions.map((r, i) => {
          const left = (r.start_sec / durationSec) * 100;
          const width = ((r.end_sec - r.start_sec) / durationSec) * 100;
          return (
            <div
              key={i}
              className="absolute top-0 h-full bg-gradient-to-r from-accent-purple to-rose-500 rounded-full"
              style={{
                left: `${left}%`,
                width: `${Math.max(width, 0.6)}%`,
                opacity: 0.4 + r.intensity * 0.6,
              }}
              title={`${r.start_sec}s - ${r.end_sec}s (intensity ${r.intensity})`}
            />
          );
        })}
      </div>

      <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
        {regions.map((r, i) => (
          <div
            key={i}
            className="flex items-center justify-between text-xs glass-card px-3 py-2"
          >
            <div className="flex items-center gap-2 text-slate-300">
              <AlertTriangle size={13} className="text-accent-purple" />
              {r.start_sec}s – {r.end_sec}s
            </div>
            <span className="text-slate-400">intensity {Math.round(r.intensity * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
