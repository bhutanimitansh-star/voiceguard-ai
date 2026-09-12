export default function StatCard({ icon: Icon, label, value, accent = "purple" }) {
  const colorMap = {
    purple: "from-accent-purple/20 to-accent-purple/5 text-accent-purple",
    cyan: "from-accent-cyan/20 to-accent-cyan/5 text-accent-cyan",
  };

  return (
    <div className="glass-card-hover p-5 flex items-center gap-4">
      <div className={`p-3 rounded-xl bg-gradient-to-br ${colorMap[accent]}`}>
        <Icon size={22} />
      </div>
      <div>
        <p className="text-2xl font-bold">{value}</p>
        <p className="text-xs text-slate-400 mt-0.5">{label}</p>
      </div>
    </div>
  );
}
