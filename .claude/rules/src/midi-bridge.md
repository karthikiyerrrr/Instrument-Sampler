# MIDI Bridge (`bridge/`)

- Use `mido` + `python-rtmidi`. Virtual port named `InstrumentSampler`.
- macOS/Linux: `virtual=True`. Windows: requires loopMIDI — open by name without `virtual=True`. Always check `sys.platform`.
- MIDI sender runs in a **dedicated thread** consuming from a queue. Never send MIDI from the audio callback thread.
- Send each event individually as it arrives. Never batch MIDI messages.
- Enumerate ports with `mido.get_output_names()`. Never hardcode port names.
