import { createServer } from "http";
import { readFile } from "fs/promises";
import { extname, join } from "path";
import { fileURLToPath } from "url";

const dir = fileURLToPath(new URL(".", import.meta.url));
const port = 8080;

const mimeTypes = {
    ".html": "text/html",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".wasm": "application/wasm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
};

createServer(async (req, res) => {
    // Required for SharedArrayBuffer (ONNX Runtime threaded WASM)
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "credentialless");
    // Allow cross-origin imports (e.g. from Scribbler)
    res.setHeader("Access-Control-Allow-Origin", "*");

    const urlPath = new URL(req.url, "http://localhost").pathname;
    if (urlPath === "/") {
        res.writeHead(404);
        res.end("Not found");
        return;
    }
    let filePath = join(dir, urlPath);
    try {
        const data = await readFile(filePath);
        res.writeHead(200, { "Content-Type": mimeTypes[extname(filePath)] || "application/octet-stream" });
        res.end(data);
    } catch {
        res.writeHead(404);
        res.end("Not found");
    }
}).listen(port, () => console.log(`http://localhost:${port}`));
