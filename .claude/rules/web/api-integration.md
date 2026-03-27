# API & WebSocket Integration

- **Strict Typing:** All interactions with the FastAPI backend must be typed. Define TypeScript interfaces for your expected JSON payloads in `web/lib/api.ts`.
- **Real-Time Data (WebSockets):** Components like `AudioMeter.tsx` and `PitchDisplay.tsx` that rely on the `api/websocket.py` stream must handle their own connection lifecycle. Always ensure WebSocket connections are closed cleanly in a `useEffect` cleanup function to prevent memory leaks when navigating away from the session page.
- **REST Fetching:** Use standard `fetch` within Server Components for initial static data. For dynamic client-side fetching (like updating a config), abstract the fetch logic into `web/lib/api.ts`.
