require("dotenv").config();
const WebSocket = require("ws");

const apiKey = process.env.SARVAM_API_KEY;

if (!apiKey) {
    console.error("SARVAM_API_KEY is missing.");
    process.exit(1);
}

const server = new WebSocket.Server({ port: 8080 });

console.log("Starting voice bridge...");
console.log("Browser WebSocket server: ws://localhost:8080");

server.on("connection", (browserWs) => {
    console.log("Browser connected.");

    // auto-detect: the corpus is Hindi/Tamil/Telugu/Bengali-only, so pinning
    // a single language (previously hardcoded en-IN) meant every real query
    // retrieved nothing. Sarvam's saaras:v3-realtime supports `auto` and
    // returns a `language` field on transcript.final when it's used --
    // App.jsx already reads and forwards that field to /v1/chat.
    const sarvamUrl =
        "wss://api.sarvam.ai/speech-to-text-realtime/ws" +
        "?language_code=auto" +
        "&sample_rate=16000";

    const sarvamWs = new WebSocket(sarvamUrl, {
        headers: {
            "Api-Subscription-Key": apiKey
        }
    });

    sarvamWs.on("open", () => {
        console.log("Connected to Sarvam Streaming STT");
        console.log("Waiting for audio...");
    });

    // React -> Node -> Sarvam
    browserWs.on("message", (data, isBinary) => {
        if (sarvamWs.readyState !== WebSocket.OPEN) {
            return;
        }

        if (isBinary) {
            // Raw PCM16 (linear16) audio frame from the browser -- forward
            // straight to Sarvam unmodified.
            sarvamWs.send(data);
            return;
        }

        try {
            // Non-binary messages are control/JSON messages from the frontend.
            const clientMessage = JSON.parse(data.toString());
            console.log("Browser control message:", clientMessage);
        } catch (error) {
            console.error("Invalid browser control message:", error.message);
        }
    });

    // Sarvam -> Node -> React
    sarvamWs.on("message", (data) => {
        const message = data.toString();

        console.log("Sarvam:", message);

        if (browserWs.readyState === WebSocket.OPEN) {
            browserWs.send(message);
        }
    });

    sarvamWs.on("error", (error) => {
        console.error("Sarvam error:", error.message);
    });

    browserWs.on("error", (error) => {
        console.error("Browser WebSocket error:", error.message);
    });

    sarvamWs.on("close", (code, reason) => {
        console.log(
            `Sarvam connection closed: ${code} ${reason.toString()}`
        );
    });

    browserWs.on("close", () => {
        console.log("Browser disconnected.");

        if (sarvamWs.readyState === WebSocket.OPEN) {
            sarvamWs.close();
        }
    });
});

server.on("error", (error) => {
    console.error("Server error:", error.message);
});