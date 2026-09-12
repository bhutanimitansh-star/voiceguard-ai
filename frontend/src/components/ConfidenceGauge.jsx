import { RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";

/**
 * Circular gauge showing overall confidence, colored by verdict.
 * AI Voice -> purple/rose gradient feel; Human -> cyan/emerald feel.
 */
export default function ConfidenceGauge({ confidence, prediction }) {
  const isAI = prediction === "AI Voice";
  const color = isAI ? "#a855f7" : "#22d3ee";
  const data = [{ name: "confidence", value: confidence, fill: color }];

  return (
    <div className="relative flex items-center justify-center">
      <RadialBarChart
        width={220}
        height={220}
        cx="50%"
        cy="50%"
        innerRadius="72%"
        outerRadius="100%"
        barSize={16}
        data={data}
        startAngle={90}
        endAngle={-270}
      >
        <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
        <RadialBar background={{ fill: "rgba(255,255,255,0.06)" }} dataKey="value" cornerRadius={12} />
      </RadialBarChart>
      <div className="absolute flex flex-col items-center">
        <span className="text-4xl font-extrabold" style={{ color }}>
          {confidence?.toFixed(1)}%
        </span>
        <span className="text-xs text-slate-400 mt-1">Confidence</span>
      </div>
    </div>
  );
}
