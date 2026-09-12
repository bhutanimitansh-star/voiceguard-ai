import { useEffect, useState } from "react";
import { Activity, Cpu } from "lucide-react";
import { fetchHealth } from "../services/api";

export default function Topbar({ title, subtitle }) {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth({ status: "offline" }));
  }, []);

  return (
    <header className="flex items-center justify-between px-6 md:px-10 py-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
        {subtitle && <p className="text-sm text-slate-400 mt-1">{subtitle}</p>}
      </div>

      <div className="flex items-center gap-3">
        <div className="glass-card px-3.5 py-2 flex items-center gap-2 text-xs">
          <Cpu size={14} className="text-accent-cyan" />
          <span className="text-slate-300">{health?.device?.toUpperCase() || "..."}</span>
        </div>
        <div className="glass-card px-3.5 py-2 flex items-center gap-2 text-xs">
          <Activity
            size={14}
            className={health?.status === "ok" ? "text-emerald-400" : "text-rose-400"}
          />
          <span className="text-slate-300">
            {health?.status === "ok" ? "System Online" : "Connecting..."}
          </span>
        </div>
      </div>
    </header>
  );
}
