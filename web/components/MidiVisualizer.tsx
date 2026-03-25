"use client";

import { useCallback, useEffect, useRef } from "react";
import type { MidiEvent } from "@/lib/types";

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function midiNoteName(note: number): string {
  return `${NOTE_NAMES[note % 12]}${Math.floor(note / 12) - 1}`;
}

interface NoteBar {
  note: number;
  velocity: number;
  startMs: number;
  endMs: number | null;
}

interface MidiVisualizerProps {
  recording: boolean;
}

const VISIBLE_DURATION_MS = 8000;
const NOTE_LO = 36;
const NOTE_HI = 96;
const NOTE_RANGE = NOTE_HI - NOTE_LO;
const LABEL_WIDTH = 44;

const BAR_HUE_BASE = 220;

export default function MidiVisualizer({ recording }: MidiVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const notesRef = useRef<NoteBar[]>([]);
  const activeNotesRef = useRef<Map<number, NoteBar>>(new Map());
  const animRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);

  const pushEvent = useCallback((event: MidiEvent) => {
    if (startTimeRef.current === 0) {
      startTimeRef.current = performance.now() - event.time_ms;
    }

    if (event.type === "note_on" && event.velocity > 0) {
      const bar: NoteBar = {
        note: event.note,
        velocity: event.velocity,
        startMs: event.time_ms,
        endMs: null,
      };
      activeNotesRef.current.set(event.note, bar);
      notesRef.current.push(bar);
    } else {
      const active = activeNotesRef.current.get(event.note);
      if (active) {
        active.endMs = event.time_ms;
        activeNotesRef.current.delete(event.note);
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      (window as unknown as Record<string, unknown>).__midiVisPush = pushEvent;
    }
    return () => {
      if (typeof window !== "undefined") {
        delete (window as unknown as Record<string, unknown>).__midiVisPush;
      }
    };
  }, [pushEvent]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, w, h);

    const drawW = w - LABEL_WIDTH;
    const rowH = h / NOTE_RANGE;

    ctx.fillStyle = "rgba(255,255,255,0.03)";
    for (let i = 0; i < NOTE_RANGE; i++) {
      const n = NOTE_LO + i;
      const isBlack = [1, 3, 6, 8, 10].includes(n % 12);
      if (isBlack) {
        ctx.fillRect(LABEL_WIDTH, h - (i + 1) * rowH, drawW, rowH);
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= NOTE_RANGE; i++) {
      const y = h - i * rowH;
      ctx.beginPath();
      ctx.moveTo(LABEL_WIDTH, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "9px var(--font-geist-mono), monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0; i < NOTE_RANGE; i++) {
      const n = NOTE_LO + i;
      if (n % 12 === 0) {
        const y = h - (i + 0.5) * rowH;
        ctx.fillText(midiNoteName(n), LABEL_WIDTH - 4, y);
      }
    }

    const nowMs = startTimeRef.current > 0
      ? performance.now() - startTimeRef.current
      : 0;
    const windowStart = nowMs - VISIBLE_DURATION_MS;

    const notes = notesRef.current;
    for (let i = notes.length - 1; i >= 0; i--) {
      const bar = notes[i];
      if (bar.endMs !== null && bar.endMs < windowStart) continue;
      if (bar.startMs > nowMs) continue;

      const noteIdx = bar.note - NOTE_LO;
      if (noteIdx < 0 || noteIdx >= NOTE_RANGE) continue;

      const endMs = bar.endMs ?? nowMs;
      const x0 = LABEL_WIDTH + ((bar.startMs - windowStart) / VISIBLE_DURATION_MS) * drawW;
      const x1 = LABEL_WIDTH + ((endMs - windowStart) / VISIBLE_DURATION_MS) * drawW;
      const y = h - (noteIdx + 1) * rowH;

      const clampedX0 = Math.max(LABEL_WIDTH, x0);
      const clampedX1 = Math.min(w, x1);
      if (clampedX1 <= clampedX0) continue;

      const hue = BAR_HUE_BASE + (bar.note % 12) * 25;
      const alpha = bar.endMs === null ? 0.9 : 0.7;
      ctx.fillStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;

      const radius = Math.min(3, rowH / 2, (clampedX1 - clampedX0) / 2);
      ctx.beginPath();
      ctx.roundRect(clampedX0, y + 1, clampedX1 - clampedX0, rowH - 2, radius);
      ctx.fill();
    }

    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(LABEL_WIDTH, 0);
    ctx.lineTo(LABEL_WIDTH, h);
    ctx.stroke();

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    if (recording) {
      animRef.current = requestAnimationFrame(draw);
    }
    return () => {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current);
        animRef.current = 0;
      }
    };
  }, [recording, draw]);

  useEffect(() => {
    if (!recording) return;
    notesRef.current = [];
    activeNotesRef.current.clear();
    startTimeRef.current = 0;
  }, [recording]);

  return (
    <div className="flex flex-col gap-2">
      <span className="text-sm font-medium text-zinc-300">
        MIDI Piano Roll
      </span>
      <div className="relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950">
        <canvas
          ref={canvasRef}
          className="h-64 w-full"
          style={{ display: "block" }}
        />
        {!recording && notesRef.current.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-zinc-600">
            Start recording to see MIDI events
          </div>
        )}
      </div>
    </div>
  );
}
