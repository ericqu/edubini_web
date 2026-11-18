# iPad microscope app: status and next steps

## Status (2026-11-11)

- Web app in `web/` works on both iPads (tested via the local server, `uv run python web/serve.py`,
  https://192.168.178.63:8443/). iPads seen at 192.168.178.188 and 192.168.178.179.
- Super photo recording: 4–8 s (default 5 s); at most 90 frames kept (every 2nd/3rd frame for long recordings).
- Rust/WebAssembly engine matches the Python pipeline (50–57 dB); parallel helper workers give the
  same result as the single-worker path; ~8 s processing on the Mac.
- The local server logs device reports (screen, cores, fps, processing time, frames, memory) to
  `results/devices.jsonl`, only when the app is served on port 8443.
- Nothing committed to git yet; no GitHub repository yet.

## Next steps

1. Start the server, take a super photo on each iPad, read `results/devices.jsonl`: decide whether
   the speed and the 90-frame cap are fine.
2. Commit, create the GitHub repository, publish `web/` (without `web/engine/`) on GitHub Pages.
   Bump `CACHE` in `web/sw.js` on every deploy.
3. On each iPad: open the GitHub Pages address → Share → Add to Home Screen (then it works offline).
4. Before switching: photos taken on the local address stay in that address's gallery. Use
   Save / share to keep any you want.
