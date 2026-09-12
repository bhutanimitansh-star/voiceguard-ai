import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, FileAudio, X } from "lucide-react";

const ACCEPTED = {
  "audio/wav": [".wav"],
  "audio/mpeg": [".mp3"],
  "audio/flac": [".flac"],
  "audio/x-m4a": [".m4a"],
  "audio/mp4": [".m4a"],
};

export default function UploadDropzone({ onFileSelected, disabled }) {
  const [selectedFile, setSelectedFile] = useState(null);

  const onDrop = useCallback(
    (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (file) {
        setSelectedFile(file);
        onFileSelected(file);
      }
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxFiles: 1,
    disabled,
  });

  const clearFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    onFileSelected(null);
  };

  return (
    <div
      {...getRootProps()}
      className={`glass-card border-2 border-dashed transition-all duration-300 p-10 text-center cursor-pointer
        ${isDragActive ? "border-accent-cyan bg-accent-cyan/5" : "border-white/15 hover:border-accent-purple/50"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <input {...getInputProps()} />

      {!selectedFile ? (
        <div className="flex flex-col items-center gap-3">
          <div className="p-4 rounded-2xl bg-gradient-to-br from-accent-purple/20 to-accent-cyan/20">
            <UploadCloud className="w-8 h-8 text-accent-cyan" />
          </div>
          <p className="font-semibold text-slate-100">
            {isDragActive ? "Drop the audio file here" : "Drag & drop an audio file"}
          </p>
          <p className="text-sm text-slate-400">or click to browse — .wav, .mp3, .flac, .m4a</p>
        </div>
      ) : (
        <div className="flex items-center justify-center gap-3">
          <FileAudio className="w-6 h-6 text-accent-purple" />
          <span className="font-medium text-slate-100">{selectedFile.name}</span>
          <span className="text-xs text-slate-400">
            ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
          </span>
          <button
            onClick={clearFile}
            className="p-1 rounded-full hover:bg-white/10 transition-colors"
          >
            <X size={16} className="text-slate-400" />
          </button>
        </div>
      )}
    </div>
  );
}
