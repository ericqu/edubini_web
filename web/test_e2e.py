"""End-to-end test of the web app in headless Chrome with a fake camera that
plays a recorded microscope video.

    uv run python web/serve.py &                         # the dev server
    uv run --with websockets python web/test_e2e.py leaf_direct.mkv

Opens the app, waits for the camera, presses "Super photo", waits for the
result and saves screenshots to results/e2e/.
"""

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import cv2
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e2e")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL = "https://localhost:8443/" + (sys.argv[2] if len(sys.argv) > 2 else "")


def make_y4m(video: str, path: str, n: int = 400) -> None:
    cap = cv2.VideoCapture(video)
    ok, f = cap.read()
    h, w = f.shape[:2]
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{w} H{h} F23:1 Ip A1:1 C420jpeg\n".encode())
        k = 0
        while ok and k < n:
            fh.write(b"FRAME\n" + cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).tobytes())
            ok, f = cap.read()
            k += 1


class CDP:
    def __init__(self, ws):
        self.ws, self.id, self.logs = ws, 0, []

    async def call(self, method, **params):
        self.id += 1
        my = self.id
        await self.ws.send(json.dumps({"id": my, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("method") == "Runtime.consoleAPICalled":
                self.logs.append(" ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"]))
            if m.get("id") == my:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m["result"]

    async def js(self, expr, await_promise=True):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=await_promise, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")

    async def shot(self, name):
        r = await self.call("Page.captureScreenshot", format="png")
        with open(os.path.join(OUT, name), "wb") as fh:
            fh.write(base64.b64decode(r["data"]))


async def run(video):
    os.makedirs(OUT, exist_ok=True)
    tmp = tempfile.mkdtemp()
    y4m = os.path.join(tmp, "cam.y4m")
    make_y4m(video, y4m)
    chrome = subprocess.Popen([
        CHROME, "--headless=new", "--remote-debugging-port=9333", f"--user-data-dir={tmp}/profile",
        "--ignore-certificate-errors", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-video-capture={y4m}", "--autoplay-policy=no-user-gesture-required",
        "--window-size=1180,820", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2**26) as ws:
            c = CDP(ws)
            await c.call("Runtime.enable")
            await c.call("Page.enable")
            await c.call("Page.navigate", url=URL)
            for _ in range(60):
                status = await c.js("document.querySelector('#status span') && document.querySelector('#status span').textContent", False)
                if status and "fps" in status:
                    break
                await asyncio.sleep(0.5)
            print("status:", status)
            await c.shot("1_live.png")
            await c.js("window.__app.takePhoto()")
            t0 = time.time()
            await c.js("window.__app.superPhoto()")
            print(f"super photo took {time.time() - t0:.1f} s (incl. {await c.js('window.__app.settings.duration', False)} s recording)")
            err = await c.js("document.getElementById('msg').hidden ? null : document.getElementById('msgText').textContent", False)
            if err:
                print("ERROR shown to user:", err)
            photos = await c.js("window.__app.allPhotos().then(l => l.map(p => ({kind: p.kind, name: p.name, w: p.w, h: p.h, bytes: p.blob.size, diag: p.diag})))")
            for p in photos:
                print(json.dumps(p))
            await asyncio.sleep(0.5)
            await c.shot("2_result.png")
            await c.js("document.getElementById('vbefore').click()", False)
            await asyncio.sleep(0.5)
            await c.shot("3_before.png")
            await c.js("document.getElementById('vclose').click(); document.getElementById('btnGallery').click()", False)
            await asyncio.sleep(0.8)
            await c.shot("4_gallery.png")
            for line in c.logs:
                print("console:", line)
    finally:
        chrome.terminate()
        chrome.wait()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "leaf_direct.mkv")))
