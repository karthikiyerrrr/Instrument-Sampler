"use client";

import { useCallback, useRef, useState } from "react";
import { connectMidiWebSocket, startSession, stopSession } from "@/lib/api";
import type { MidiEvent, StopResponse } from "@/lib/types";

interface SessionControlsProps {
  deviceIndex: number | null;
  onMidiEvent: (event: MidiEvent) => void;
  onSessionStart: () => void;
  onSessionStop: (info: StopResponse) => void;
}

export default function SessionControls({
  deviceIndex,
  onMidiEvent,
  onSessionStart,
  onSessionStop,
}: SessionControlsProps) {
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const handleStart = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      await startSession(deviceIndex);
      wsRef.current = connectMidiWebSocket(
        (event) => onMidiEvent(event),
        () => {
          wsRef.current = null;
        },
      );
      setRecording(true);
      onSessionStart();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [deviceIndex, onMidiEvent, onSessionStart]);

  const handleStop = useCallback(async () => {
    setError(null);
    setBusy(true);
    try {
      wsRef.current?.close();
      wsRef.current = null;
      const info = await stopSession();
      setRecording(false);
      onSessionStop(info);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [onSessionStop]);

  return (
    <div className="flex flex-col items-center gap-3">
      <button
        disabled={busy}
        onClick={recording ? handleStop : handleStart}
        className={`relative flex h-14 w-14 items-center justify-center rounded-full border-2 transition-all disabled:opacity-50 ${
          recording
            ? "border-red-500 bg-red-500/20 hover:bg-red-500/30"
            : "border-zinc-600 bg-zinc-800 hover:border-indigo-500 hover:bg-zinc-700"
        }`}
        aria-label={recording ? "Stop recording" : "Start recording"}
      >
        {recording ? (
          <>
            <span className="absolute inset-0 animate-ping rounded-full bg-red-500/30" />
            <span className="relative h-5 w-5 rounded-sm bg-red-500" />
          </>
        ) : (
          <span className="relative h-5 w-5 rounded-full bg-red-500" />
        )}
      </button>

      <span className="text-xs font-medium text-zinc-400">
        {busy ? "..." : recording ? "Recording" : "Ready"}
      </span>

      {error && (
        <span className="max-w-xs text-center text-xs text-red-400">
          {error}
        </span>
      )}
    </div>
  );
}
