import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

export default function ProbabilityBars({ humanProbability, aiProbability }) {
  const data = [
    { name: "Human", value: humanProbability, fill: "#22d3ee" },
    { name: "AI Voice", value: aiProbability, fill: "#a855f7" },
  ];

  return (
    <div className="w-full h-40">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis
            type="category"
            dataKey="name"
            width={80}
            tick={{ fill: "#cbd5e1", fontSize: 13, fontWeight: 500 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            contentStyle={{
              background: "#161625",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 12,
              color: "white",
            }}
            formatter={(value) => [`${value.toFixed(2)}%`, "Probability"]}
          />
          <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={26}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
