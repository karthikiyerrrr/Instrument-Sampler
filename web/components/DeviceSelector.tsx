"use client";

import { useEffect, useState } from "react";
import { getDevices } from "@/lib/api";
import type { AudioDevice } from "@/lib/types";

interface DeviceSelectorProps {
  selectedDevice: number | null;
  onSelect: (deviceIndex: number | null) => void;
  disabled?: boolean;
}

export default function DeviceSelector({
  selectedDevice,
  onSelect,
  disabled = false,
}: DeviceSelectorProps) {
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDevices()
      .then((devs) => {
        setDevices(devs);
        if (devs.length > 0 && selectedDevice === null) {
          onSelect(devs[0].index);
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
    // Only run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return (
      <div className="text-red-400 text-sm">
        Failed to load devices: {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="device-select" className="text-sm font-medium text-zinc-300">
        Input Device
      </label>
      {loading ? (
        <div className="h-10 w-full animate-pulse rounded-lg bg-zinc-800" />
      ) : (
        <select
          id="device-select"
          className="h-10 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 text-sm text-zinc-100 outline-none transition focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
          value={selectedDevice ?? ""}
          disabled={disabled}
          onChange={(e) => {
            const val = e.target.value;
            onSelect(val === "" ? null : Number(val));
          }}
        >
          <option value="">System Default</option>
          {devices.map((d) => (
            <option key={d.index} value={d.index}>
              {d.name} ({d.max_input_channels}ch, {d.default_samplerate} Hz)
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
