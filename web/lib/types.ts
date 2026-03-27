export interface AudioDevice {
  index: number;
  name: string;
  max_input_channels: number;
  default_samplerate: number;
}

export interface SessionStatus {
  status: "idle" | "recording";
  wav_path?: string;
}

export interface StartResponse {
  status: "recording";
  wav_path: string;
}

export interface StopResponse {
  status: "stopped";
  wav_path: string;
  duration_s: number;
}

export interface MidiEvent {
  type: "note_on" | "note_off";
  note: number;
  velocity: number;
  time_ms: number;
}

// Post-processing

export interface TranscribeResponse {
  status: string;
  midi_path: string;
}

export interface CalibrateResponse {
  status: string; // "queued" | "done" | "error"
  job_id: string;
  model_dir?: string;
  error?: string;
}

export interface CloneResponse {
  status: string;
  cloned_wav_path: string;
}

export interface ModelInfo {
  name: string;
  model_dir: string;
  has_onnx: boolean;
}
