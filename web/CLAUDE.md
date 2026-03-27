# web/

Next.js 16 frontend — browser-based dashboard for device selection, session control, and live MIDI visualization. Rules for frontend development are in `.claude/rules/web/`.

## Structure

- `app/` — App Router pages (`layout.tsx`, `page.tsx`)
- `components/` — Client components (`Dashboard.tsx`, `DeviceSelector.tsx`, `SessionControls.tsx`, `MidiVisualizer.tsx`)
- `lib/` — Typed fetch helpers (`api.ts`) and shared TypeScript interfaces (`types.ts`)
- `public/` — Static assets

## Dependencies

Install from `web/`: `npm install`.

| Package | Purpose |
|---------|---------|
| `next` (v16) | App Router framework |
| `react` / `react-dom` | UI rendering |
| `typescript` | Type safety |
| `tailwindcss` | Utility-first styling |
| `eslint` | Linting |

Requires Node.js 18+.

---

## Pending Tasks

No pending tasks.
