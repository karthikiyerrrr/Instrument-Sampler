"use client";

import { useCallback, useState } from "react";
import DeviceSelector from "@/components/DeviceSelector";
import MidiVisualizer from "@/components/MidiVisualizer";
import SessionControls from "@/components/SessionControls";
import type { MidiEvent, StopResponse } from "@/lib/types";

export default function Dashboard() {
  const [deviceIndex, setDeviceIndex] = useState<number | null>(null);
  const [recording, setRecording] = useState(false);
  const [lastSession, setLastSession] = useState<StopResponse | null>(null);

  const handleMidiEvent = useCallback((event: MidiEvent) => {
    const push = (
      window as unknown as Record<string, (e: MidiEvent) => void>
    ).__midiVisPush;
    if (push) push(event);
  }, []);

  const handleSessionStart = useCallback(() => {
    setRecording(true);
    setLastSession(null);
  }, []);

  const handleSessionStop = useCallback((info: StopResponse) => {
    setRecording(false);
    setLastSession(info);
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-6 py-10">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-zinc-100">
          Instrument Sampler
        </h1>
        <p className="text-sm text-zinc-500">
          Live audio capture with real-time MIDI feedback
        </p>
      </header>

      <div className="flex flex-col gap-6 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
          <div className="flex-1">
            <DeviceSelector
              selectedDevice={deviceIndex}
              onSelect={setDeviceIndex}
              disabled={recording}
            />
          </div>
          <SessionControls
            deviceIndex={deviceIndex}
            onMidiEvent={handleMidiEvent}
            onSessionStart={handleSessionStart}
            onSessionStop={handleSessionStop}
          />
        </div>

        <MidiVisualizer recording={recording} />

        {lastSession && (
          <div className="flex items-center gap-4 rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-3 text-sm text-zinc-400">
            <span>
              Session saved: <span className="font-mono text-zinc-300">{lastSession.wav_path}</span>
            </span>
            <span className="text-zinc-600">|</span>
            <span>{lastSession.duration_s}s</span>
          </div>
        )}
      </div>
    </div>
  );
}
