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
