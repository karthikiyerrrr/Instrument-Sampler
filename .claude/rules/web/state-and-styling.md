# State Management & Styling

## State Management

- Avoid heavy global state libraries (like Redux) unless strictly necessary. Use standard React state for component-level UI toggles.
- For shared live data (e.g., the current session state or active instrument config), consider React Context wrapping only the specific interactive sub-trees, rather than the whole app.

## Styling & UI

- **Tailwind CSS:** Use Tailwind utility classes for all styling. Avoid creating custom `.css` files unless overriding a complex third-party component.
- **Responsive Design:** Ensure dashboards and diagnostic panels use flexbox/grid to scale cleanly, as audio routing configurations can take up significant screen real estate.
