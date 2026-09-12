import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ShieldCheck, Zap, Database, TrendingUp, AudioWaveform } from "lucide-react";
import Topbar from "../components/Topbar";
import StatCard from "../components/StatCard";
import { fetchHistory } from "../services/api";
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  LineChart, Line, XAxis, YAxis, CartesianGrid,
} from "recharts";

export default function Dashboard() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory(50)
      .then((data) => setHistory(data.items))
      .catch(() => setHistory([]))
      .finally(() => setLoading(false));
  }, []);

  const total = history.length;
  const aiCount = history.filter((h) => h.prediction === "AI Voice").length;
  const humanCount = total - aiCount;
  const avgConfidence = total
    ? (history.reduce((sum, h) => sum + h.confidence, 0) / total).toFixed(1)
    : "—";

  const pieData = [
    { name: "Human", value: humanCount, color: "#22d3ee" },
    { name: "AI Voice", value: aiCount, color: "#a855f7" },
  ];

  const trendData = [...history]
    .reverse()
    .slice(-20)
    .map((h, i) => ({ index: i + 1, confidence: h.confidence }));

  return (
    <div>
      <Topbar
        title="Dashboard"
        subtitle="Overview of your VoiceGuard AI detection activity"
      />

      <div className="px-6 md:px-10 pb-10 space-y-8">
        {/* Stat cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          <StatCard icon={Database} label="Total Scans" value={total} accent="cyan" />
          <StatCard icon={ShieldCheck} label="Human Voices" value={humanCount} accent="cyan" />
          <StatCard icon={AudioWaveform} label="AI Voices Detected" value={aiCount} accent="purple" />
          <StatCard icon={TrendingUp} label="Avg. Confidence" value={`${avgConfidence}%`} accent="purple" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* CTA card */}
          <div className="lg:col-span-1 glass-card-hover p-7 flex flex-col justify-between bg-gradient-to-br from-accent-purple/10 to-accent-cyan/10">
            <div>
              <div className="p-3 w-fit rounded-xl bg-white/10 mb-4">
                <Zap className="text-accent-cyan" size={22} />
              </div>
              <h3 className="text-xl font-bold mb-2">Run a New Analysis</h3>
              <p className="text-sm text-slate-400 leading-relaxed">
                Upload a clip and get a Human vs AI verdict with confidence
                scores, spectrograms, and explainable attention highlights
                in under 3 seconds.
              </p>
            </div>
            <Link to="/analyzer" className="btn-primary text-center mt-6">
              Analyze Audio →
            </Link>
          </div>

          {/* Pie chart */}
          <div className="glass-card p-6">
            <h3 className="font-semibold mb-4">Verdict Distribution</h3>
            {total === 0 ? (
              <EmptyState />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={4}
                  >
                    {pieData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: "#161625", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12 }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}
            <div className="flex justify-center gap-6 mt-2 text-xs">
              <span className="flex items-center gap-1.5 text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full bg-accent-cyan" /> Human
              </span>
              <span className="flex items-center gap-1.5 text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full bg-accent-purple" /> AI Voice
              </span>
            </div>
          </div>

          {/* Trend chart */}
          <div className="glass-card p-6">
            <h3 className="font-semibold mb-4">Recent Confidence Trend</h3>
            {total === 0 ? (
              <EmptyState />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={trendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="index" tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} axisLine={false} />
                  <Tooltip
                    contentStyle={{ background: "#161625", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="confidence"
                    stroke="#a855f7"
                    strokeWidth={2.5}
                    dot={{ fill: "#22d3ee", r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="h-[220px] flex items-center justify-center text-sm text-slate-500">
      No scans yet — run your first analysis.
    </div>
  );
}
