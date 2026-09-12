import { useEffect, useState } from "react";
import { Trash2, FileAudio, RefreshCw } from "lucide-react";
import Topbar from "../components/Topbar";
import { fetchHistory, clearHistory } from "../services/api";

export default function History() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    fetchHistory(200)
      .then((data) => setItems(data.items))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const handleClear = async () => {
    if (!confirm("Clear all prediction history? This cannot be undone.")) return;
    await clearHistory();
    load();
  };

  return (
    <div>
      <Topbar title="History" subtitle="All past predictions stored in SQLite" />

      <div className="px-6 md:px-10 pb-10">
        <div className="flex items-center justify-between mb-5">
          <p className="text-sm text-slate-400">{items.length} total records</p>
          <div className="flex gap-3">
            <button onClick={load} className="btn-secondary flex items-center gap-2 text-sm">
              <RefreshCw size={15} /> Refresh
            </button>
            <button
              onClick={handleClear}
              className="btn-secondary flex items-center gap-2 text-sm text-rose-300 hover:text-rose-200"
            >
              <Trash2 size={15} /> Clear History
            </button>
          </div>
        </div>

        <div className="glass-card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-white/10">
                <th className="px-5 py-3 font-medium">File</th>
                <th className="px-5 py-3 font-medium">Verdict</th>
                <th className="px-5 py-3 font-medium">Confidence</th>
                <th className="px-5 py-3 font-medium">Human %</th>
                <th className="px-5 py-3 font-medium">AI %</th>
                <th className="px-5 py-3 font-medium">Time</th>
                <th className="px-5 py-3 font-medium">Date</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} className="text-center py-10 text-slate-500">
                    Loading...
                  </td>
                </tr>
              ) : items.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-10 text-slate-500">
                    No predictions yet.
                  </td>
                </tr>
              ) : (
                items.map((item) => (
                  <tr
                    key={item.id}
                    className="border-b border-white/5 hover:bg-white/5 transition-colors"
                  >
                    <td className="px-5 py-3 flex items-center gap-2">
                      <FileAudio size={14} className="text-accent-cyan shrink-0" />
                      <span className="truncate max-w-[160px]">{item.filename}</span>
                    </td>
                    <td className="px-5 py-3">
                      <span className={item.prediction === "AI Voice" ? "pill-ai" : "pill-human"}>
                        {item.prediction}
                      </span>
                    </td>
                    <td className="px-5 py-3 font-semibold">{item.confidence}%</td>
                    <td className="px-5 py-3 text-accent-cyan">{item.human_probability}%</td>
                    <td className="px-5 py-3 text-accent-purple">{item.ai_probability}%</td>
                    <td className="px-5 py-3 text-slate-400">{item.processing_time}</td>
                    <td className="px-5 py-3 text-slate-400">
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
