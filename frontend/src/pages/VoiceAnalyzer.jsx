import { useState } from "react";
import { Loader2, ShieldAlert, ShieldCheck, Clock, Waves } from "lucide-react";
import Topbar from "../components/Topbar";
import UploadDropzone from "../components/UploadDropzone";
import ConfidenceGauge from "../components/ConfidenceGauge";
import ProbabilityBars from "../components/ProbabilityBars";
import SuspiciousTimeline from "../components/SuspiciousTimeline";
import { predictVoice } from "../services/api";

export default function VoiceAnalyzer() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predictVoice(file, (evt) => {
        setProgress(Math.round((evt.loaded * 100) / evt.total));
      });
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  const isAI = result?.prediction === "AI Voice";

  return (
    <div>
      <Topbar
        title="Voice Analyzer"
        subtitle="Upload an audio clip to detect Human vs AI-generated speech"
      />

      <div className="px-6 md:px-10 pb-10 space-y-8">
        <UploadDropzone onFileSelected={setFile} disabled={loading} />

        <div className="flex items-center gap-4">
          <button
            className="btn-primary flex items-center gap-2"
            disabled={!file || loading}
            onClick={handleAnalyze}
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={18} />
                Analyzing {progress > 0 && `(${progress}%)`}
              </>
            ) : (
              <>
                <Waves size={18} />
                Run Detection
              </>
            )}
          </button>
          {file && !loading && (
            <span className="text-sm text-slate-400">Ready to analyze “{file.name}”</span>
          )}
        </div>

        {error && (
          <div className="glass-card p-4 border-rose-500/40 text-rose-300 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-6 animate-in fade-in duration-500">
            {/* Verdict banner */}
            <div
              className={`glass-card-hover p-6 flex flex-col md:flex-row items-center justify-between gap-6 ${
                isAI ? "border-rose-500/30" : "border-emerald-500/30"
              }`}
            >
              <div className="flex items-center gap-4">
                <div
                  className={`p-4 rounded-2xl ${
                    isAI ? "bg-rose-500/15" : "bg-emerald-500/15"
                  }`}
                >
                  {isAI ? (
                    <ShieldAlert className="text-rose-400" size={30} />
                  ) : (
                    <ShieldCheck className="text-emerald-400" size={30} />
                  )}
                </div>
                <div>
                  <p className="text-sm text-slate-400">Verdict</p>
                  <h3 className="text-2xl font-extrabold">{result.prediction}</h3>
                </div>
                <span className={isAI ? "pill-ai" : "pill-human"}>
                  {isAI ? "Synthetic Voice" : "Natural Voice"}
                </span>
              </div>

              <div className="flex items-center gap-6 text-sm text-slate-300">
                <div className="flex items-center gap-2">
                  <Clock size={15} className="text-accent-cyan" />
                  {result.processing_time}
                </div>
                <div className="flex items-center gap-2">
                  <Waves size={15} className="text-accent-purple" />
                  {result.duration_sec}s clip
                </div>
              </div>
            </div>

            {/* Gauge + probability bars */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass-card p-6 flex items-center justify-center">
                <ConfidenceGauge confidence={result.confidence} prediction={result.prediction} />
              </div>
              <div className="glass-card p-6">
                <h3 className="font-semibold mb-4">Class Probabilities</h3>
                <ProbabilityBars
                  humanProbability={result.human_probability}
                  aiProbability={result.ai_probability}
                />
              </div>
            </div>

            {/* Waveform */}
            <div className="glass-card p-6">
              <h3 className="font-semibold mb-4">Waveform (suspicious regions highlighted)</h3>
              <img
                src={`data:image/png;base64,${result.waveform_png_base64}`}
                alt="Waveform"
                className="w-full rounded-xl"
              />
              <div className="mt-4">
                <SuspiciousTimeline
                  regions={result.suspicious_regions}
                  durationSec={result.duration_sec}
                />
              </div>
            </div>

            {/* Spectrogram + Grad-CAM */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="glass-card p-6">
                <h3 className="font-semibold mb-4">Mel Spectrogram</h3>
                <img
                  src={`data:image/png;base64,${result.melspectrogram_png_base64}`}
                  alt="Mel Spectrogram"
                  className="w-full rounded-xl"
                />
              </div>
              <div className="glass-card p-6">
                <h3 className="font-semibold mb-4">Grad-CAM Attention Overlay</h3>
                <img
                  src={`data:image/png;base64,${result.gradcam_png_base64}`}
                  alt="Grad-CAM heatmap"
                  className="w-full rounded-xl"
                />
              </div>
            </div>

            {/* Acoustic features */}
            <div className="glass-card p-6">
              <h3 className="font-semibold mb-4">Acoustic Feature Summary</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                {Object.entries(result.acoustic_features).map(([key, value]) => (
                  <div key={key} className="glass-card p-3 text-center">
                    <p className="text-lg font-bold text-accent-cyan">
                      {typeof value === "number" ? value.toFixed(2) : value}
                    </p>
                    <p className="text-[11px] text-slate-400 mt-1 capitalize">
                      {key.replaceAll("_", " ")}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
