# Next.js Component Rules

> This is NOT the Next.js you know. This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.

## Server vs. Client Components (App Router)

- **Default to Server Components:** All files in `web/app/` are Server Components by default. Keep them that way to handle initial data fetching (e.g., listing past recordings) without sending unnecessary JavaScript to the browser.
- **Isolate Interactivity:** Only use the `"use client"` directive at the absolute top of a file when you need React hooks (`useState`, `useEffect`), browser APIs (Web Audio API), or WebSockets.
- **Push Client Components Down:** Keep client components as leaves in your component tree. For example, `web/app/session/page.tsx` can be a Server Component that imports a `<SessionControls />` Client Component.
