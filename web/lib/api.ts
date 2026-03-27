import type {
  AudioDevice,
  CalibrateResponse,
  CloneResponse,
  MidiEvent,
  ModelInfo,
  SessionStatus,
  StartResponse,
  StopResponse,
  TranscribeResponse,
} from "./types";

const BASE = "";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export function getDevices(): Promise<AudioDevice[]> {
  return json<AudioDevice[]>("/api/devices");
}

export function startSession(deviceIndex: number | null): Promise<StartResponse> {
  return json<StartResponse>("/api/session/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_index: deviceIndex }),
  });
}

export function stopSession(): Promise<StopResponse> {
  return json<StopResponse>("/api/session/stop", {
    method: "POST",
  });
}

export function getSessionStatus(): Promise<SessionStatus> {
  return json<SessionStatus>("/api/session/status");
}

// Post-processing

export function transcribeWav(wavPath: string): Promise<TranscribeResponse> {
  return json<TranscribeResponse>("/api/post-process/transcribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ wav_path: wavPath }),
  });
}

export function startCalibration(
  wavPath: string,
  modelName: string,
  steps = 300,
): Promise<CalibrateResponse> {
  return json<CalibrateResponse>("/api/post-process/calibrate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ wav_path: wavPath, model_name: modelName, steps }),
  });
}

export function pollCalibration(jobId: string): Promise<CalibrateResponse> {
  return json<CalibrateResponse>(`/api/post-process/calibrate/${jobId}`);
}

export function cloneTimbre(
  wavPath: string,
  modelDir: string,
): Promise<CloneResponse> {
  return json<CloneResponse>("/api/post-process/clone", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ wav_path: wavPath, model_dir: modelDir }),
  });
}

export function listModels(): Promise<ModelInfo[]> {
  return json<ModelInfo[]>("/api/post-process/models");
}

export function connectMidiWebSocket(
  onEvent: (event: MidiEvent) => void,
  onClose?: () => void,
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/midi`);

  ws.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as MidiEvent;
      onEvent(event);
    } catch {
      // ignore malformed messages
    }
  };

  ws.onclose = () => {
    onClose?.();
  };

  return ws;
}
