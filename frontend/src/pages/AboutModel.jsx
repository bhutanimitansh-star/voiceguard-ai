import { Layers, Brain, Gauge, GitBranch, Sparkles } from "lucide-react";
import Topbar from "../components/Topbar";

const pipelineStages = [
  { title: "Audio Ingestion", detail: "Accepts .wav/.mp3/.flac/.m4a, decoded via ffmpeg/pydub" },
  { title: "Resampling", detail: "Converted to 16kHz mono PCM" },
  { title: "VAD Silence Removal", detail: "WebRTC VAD trims non-speech segments" },
  { title: "Denoising & Loudness Norm", detail: "Spectral gating + RMS loudness normalization" },
  { title: "Windowing", detail: "Segmented into fixed 4s windows with 50% overlap" },
  { title: "Feature Extraction", detail: "Mel Spectrogram, MFCC(40), Chroma, Centroid, Rolloff, ZCR, F0, HNR" },
  { title: "CNN Encoder", detail: "Residual Conv2D blocks with BatchNorm + Dropout" },
  { title: "BiLSTM", detail: "2-layer bidirectional LSTM over the temporal axis" },
  { title: "Attention", detail: "Additive attention pools context + yields explainability weights" },
  { title: "Classification Head", detail: "Dense fusion with acoustic stats -> Softmax(Human, AI Voice)" },
];

const metrics = [
  { label: "Precision", value: "—", note: "Populated after training run" },
  { label: "Recall", value: "—", note: "Populated after training run" },
  { label: "F1 Score", value: "—", note: "Populated after training run" },
  { label: "ROC-AUC", value: "—", note: "Populated after training run" },
];

export default function AboutModel() {
  return (
    <div>
      <Topbar title="About the Model" subtitle="Architecture, pipeline, and explainability internals" />

      <div className="px-6 md:px-10 pb-10 space-y-8">
        <div className="glass-card p-7">
          <div className="flex items-center gap-3 mb-3">
            <Brain className="text-accent-purple" size={22} />
            <h3 className="text-xl font-bold">CNN + BiLSTM + Attention</h3>
          </div>
          <p className="text-slate-400 text-sm leading-relaxed max-w-3xl">
            VoiceGuard AI classifies audio as <span className="text-emerald-300 font-medium">Human</span> or{" "}
            <span className="text-rose-300 font-medium">AI-generated</span> using a hybrid deep
            architecture. A residual convolutional encoder learns local
            time-frequency textures (formants, vocoder artifacts) from the
            log-Mel spectrogram; a bidirectional LSTM then models long-range
            temporal dependencies such as unnatural prosody; and an additive
            attention layer pools the sequence into a single context vector
            while exposing per-timestep importance for explainability.
          </p>
        </div>

        <div className="glass-card p-7">
          <div className="flex items-center gap-3 mb-5">
            <Layers className="text-accent-cyan" size={22} />
            <h3 className="text-xl font-bold">Processing Pipeline</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {pipelineStages.map((stage, i) => (
              <div key={i} className="flex gap-3 glass-card p-4">
                <div className="w-7 h-7 shrink-0 rounded-full bg-gradient-to-br from-accent-purple to-accent-cyan flex items-center justify-center text-xs font-bold">
                  {i + 1}
                </div>
                <div>
                  <p className="font-semibold text-sm">{stage.title}</p>
                  <p className="text-xs text-slate-400 mt-0.5">{stage.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="glass-card p-7">
            <div className="flex items-center gap-3 mb-4">
              <Gauge className="text-accent-purple" size={22} />
              <h3 className="text-xl font-bold">Evaluation Metrics</h3>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Filled in automatically by <code className="text-accent-cyan">training/train.py</code> after
              a full K-fold cross-validated run on your dataset.
            </p>
            <div className="grid grid-cols-2 gap-4">
              {metrics.map((m) => (
                <div key={m.label} className="glass-card p-4 text-center">
                  <p className="text-2xl font-bold text-accent-cyan">{m.value}</p>
                  <p className="text-xs text-slate-400 mt-1">{m.label}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-card p-7">
            <div className="flex items-center gap-3 mb-4">
              <Sparkles className="text-accent-cyan" size={22} />
              <h3 className="text-xl font-bold">Explainable AI</h3>
            </div>
            <ul className="space-y-3 text-sm text-slate-400">
              <li className="flex gap-2">
                <GitBranch size={15} className="text-accent-purple shrink-0 mt-0.5" />
                <span>
                  <span className="text-slate-200 font-medium">Attention timeline:</span> per-timestep
                  BiLSTM attention weights are upsampled and overlaid on the waveform to highlight
                  suspicious regions.
                </span>
              </li>
              <li className="flex gap-2">
                <GitBranch size={15} className="text-accent-purple shrink-0 mt-0.5" />
                <span>
                  <span className="text-slate-200 font-medium">Grad-CAM:</span> gradient-weighted
                  activation maps from the final residual conv block reveal which
                  time-frequency regions of the Mel spectrogram drove the verdict.
                </span>
              </li>
            </ul>
          </div>
        </div>

        <div className="glass-card p-7">
          <h3 className="text-xl font-bold mb-3">Training Datasets</h3>
          <div className="flex flex-wrap gap-2">
            {["Fake-or-Real Voice Dataset", "ASVspoof", "LibriSpeech (human)", "ElevenLabs / TTS synthetic voices"].map(
              (d) => (
                <span key={d} className="pill bg-white/5 border border-white/10 text-slate-300">
                  {d}
                </span>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
