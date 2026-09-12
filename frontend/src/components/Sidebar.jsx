import { NavLink } from "react-router-dom";
import { LayoutDashboard, AudioWaveform, History, Info, ShieldCheck } from "lucide-react";

const links = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analyzer", label: "Voice Analyzer", icon: AudioWaveform },
  { to: "/history", label: "History", icon: History },
  { to: "/about", label: "About Model", icon: Info },
];

export default function Sidebar() {
  return (
    <aside className="hidden md:flex flex-col w-64 shrink-0 h-screen sticky top-0 p-5 border-r border-white/10">
      <div className="flex items-center gap-2 mb-10 px-1">
        <div className="p-2 rounded-xl bg-gradient-to-br from-accent-purple to-accent-cyan shadow-glow">
          <ShieldCheck className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold leading-tight">VoiceGuard</h1>
          <p className="text-xs text-slate-400 -mt-0.5">AI Voice Detection</p>
        </div>
      </div>

      <nav className="flex flex-col gap-1.5">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? "bg-white/10 text-white border border-accent-purple/40 shadow-glow"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            <Icon className="w-4.5 h-4.5" size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto glass-card p-4">
        <p className="text-xs text-slate-400 leading-relaxed">
          CNN + BiLSTM + Attention network for explainable synthetic voice
          detection.
        </p>
        <p className="text-[10px] text-slate-500 mt-2">v1.0.0 · PyTorch + FastAPI</p>
      </div>
    </aside>
  );
}
