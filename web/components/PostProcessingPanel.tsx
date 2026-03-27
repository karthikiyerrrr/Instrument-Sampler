"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cloneTimbre,
  listModels,
  pollCalibration,
  startCalibration,
  transcribeWav,
} from "@/lib/api";
import type {
  CalibrateResponse,
  CloneResponse,
  ModelInfo,
  TranscribeResponse,
} from "@/lib/types";

interface Props {
  wavPath: string;
}

type Status = "idle" | "busy" | "done" | "error";

export default function PostProcessingPanel({ wavPath }: Props) {
  // --- Transcription ---
  const [txStatus, setTxStatus] = useState<Status>("idle");
  const [txResult, setTxResult] = useState<TranscribeResponse | null>(null);
  const [txError, setTxError] = useState<string | null>(null);

  const handleTranscribe = useCallback(async () => {
    setTxStatus("busy");
    setTxError(null);
    try {
      const res = await transcribeWav(wavPath);
      setTxResult(res);
      setTxStatus("done");
    } catch (err) {
      setTxError(String(err));
      setTxStatus("error");
    }
  }, [wavPath]);

  // --- Calibration ---
  const [calStatus, setCalStatus] = useState<Status>("idle");
  const [calResult, setCalResult] = useState<CalibrateResponse | null>(null);
  const [calError, setCalError] = useState<string | null>(null);
  const [modelName, setModelName] = useState("my_instrument");
  const pollRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleCalibrate = useCallback(async () => {
    setCalStatus("busy");
    setCalError(null);
    setCalResult(null);
    try {
      const job = await startCalibration(wavPath, modelName);
      setCalResult(job);

      // Poll until done or error.
      const poll = async () => {
        try {
          const latest = await pollCalibration(job.job_id);
          setCalResult(latest);
          if (latest.status === "queued") {
            pollRef.current = setTimeout(poll, 3000);
          } else if (latest.status === "done") {
            setCalStatus("done");
            loadModels();
          } else {
            setCalError(latest.error ?? "Unknown error");
            setCalStatus("error");
          }
        } catch (err) {
          setCalError(String(err));
          setCalStatus("error");
        }
      };
      pollRef.current = setTimeout(poll, 3000);
    } catch (err) {
      setCalError(String(err));
      setCalStatus("error");
    }
  }, [wavPath, modelName]);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearTimeout(pollRef.current);
    };
  }, []);

  // --- Timbre cloning ---
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [cloneStatus, setCloneStatus] = useState<Status>("idle");
  const [cloneResult, setCloneResult] = useState<CloneResponse | null>(null);
  const [cloneError, setCloneError] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    try {
      const list = await listModels();
      setModels(list);
      if (list.length > 0 && !selectedModel) {
        setSelectedModel(list[0].model_dir);
      }
    } catch {
      // non-fatal — models list just stays empty
    }
  }, [selectedModel]);

  useEffect(() => {
    loadModels();
  }, []);

  const handleClone = useCallback(async () => {
    if (!selectedModel) return;
    setCloneStatus("busy");
    setCloneError(null);
    try {
      const res = await cloneTimbre(wavPath, selectedModel);
      setCloneResult(res);
      setCloneStatus("done");
    } catch (err) {
      setCloneError(String(err));
      setCloneStatus("error");
    }
  }, [wavPath, selectedModel]);

  return (
    <div className="flex flex-col gap-6 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6">
      <h2 className="text-sm font-semibold uppercase tracking-widest text-zinc-500">
        Post-Processing
      </h2>

      {/* Transcription */}
      <Section label="Polyphonic Transcription (basic-pitch)">
        <button
          onClick={handleTranscribe}
          disabled={txStatus === "busy"}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-40"
        >
          {txStatus === "busy" ? "Transcribing…" : "Transcribe to MIDI"}
        </button>
        {txStatus === "done" && txResult && (
          <p className="font-mono text-xs text-zinc-300">
            {txResult.midi_path}
          </p>
        )}
        {txStatus === "error" && (
          <p className="text-xs text-red-400">{txError}</p>
        )}
      </Section>

      {/* Calibration */}
      <Section label="DDSP Calibration (fine-tune timbre model)">
        <div className="flex gap-2">
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            placeholder="Model name"
            className="flex-1 rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          <button
            onClick={handleCalibrate}
            disabled={calStatus === "busy" || !modelName.trim()}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-40"
          >
            {calStatus === "busy" ? "Training…" : "Calibrate"}
          </button>
        </div>
        {calStatus === "busy" && calResult && (
          <p className="text-xs text-zinc-400">
            Job <span className="font-mono">{calResult.job_id}</span> — training
            in background…
          </p>
        )}
        {calStatus === "done" && calResult && (
          <p className="font-mono text-xs text-zinc-300">{calResult.model_dir}</p>
        )}
        {calStatus === "error" && (
          <p className="text-xs text-red-400">{calError}</p>
        )}
      </Section>

      {/* Timbre cloning */}
      <Section label="Timbre Cloning (DDSP inference)">
        {models.length === 0 ? (
          <p className="text-xs text-zinc-500">
            No models found — run calibration first.
          </p>
        ) : (
          <div className="flex gap-2">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="flex-1 rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              {models.map((m) => (
                <option key={m.model_dir} value={m.model_dir} disabled={!m.has_onnx}>
                  {m.name}{m.has_onnx ? "" : " (not ready)"}
                </option>
              ))}
            </select>
            <button
              onClick={handleClone}
              disabled={cloneStatus === "busy" || !selectedModel}
              className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white hover:bg-violet-500 disabled:opacity-40"
            >
              {cloneStatus === "busy" ? "Cloning…" : "Clone Timbre"}
            </button>
          </div>
        )}
        {cloneStatus === "done" && cloneResult && (
          <p className="font-mono text-xs text-zinc-300">
            {cloneResult.cloned_wav_path}
          </p>
        )}
        {cloneStatus === "error" && (
          <p className="text-xs text-red-400">{cloneError}</p>
        )}
      </Section>
    </div>
  );
}

function Section({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs font-medium text-zinc-400">{label}</p>
      {children}
    </div>
  );
}
