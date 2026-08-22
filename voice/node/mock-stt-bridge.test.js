/*
 * TEST-ONLY stand-in for server.js, used because this sandbox has no
 * SARVAM_API_KEY / real microphone. Mimics the exact realtime Sarvam STT
 * message contract server.js forwards verbatim (event: "transcript.partial"
 * / "transcript.final", field "text", optional "language"), so it exercises
 * the same App.jsx parsing code path server.js would in production.
 *
 * Not part of the shipped app -- not referenced by package.json, not wired
 * into the real startup sequence.
 */
const WebSocket = require("ws");

const FINAL_TEXT = process.env.MOCK_FINAL_TEXT || "What is the capital of India?";
const FINAL_LANGUAGE = process.env.MOCK_FINAL_LANGUAGE || null;
const PORT = Number(process.env.MOCK_PORT || 8080);

const server = new WebSocket.Server({ port: PORT });
console.log(`[mock-stt-bridge] listening on ws://localhost:${PORT}, will emit: ${JSON.stringify({ text: FINAL_TEXT, language: FINAL_LANGUAGE })}`);

server.on("connection", (browserWs) => {
  console.log("[mock-stt-bridge] browser connected");
  let sentFinal = false;

  browserWs.on("message", () => {
    // Ignore incoming audio chunks; after a couple of them, emit a partial
    // then a final transcript, same as a real Sarvam session would.
    if (sentFinal) return;
    sentFinal = true;

    setTimeout(() => {
      browserWs.send(JSON.stringify({
        event: "transcript.partial",
        utterance_idx: 0,
        text: FINAL_TEXT.slice(0, Math.max(3, Math.floor(FINAL_TEXT.length / 2))),
      }));
    }, 150);

    setTimeout(() => {
      const payload = { event: "transcript.final", utterance_idx: 0, text: FINAL_TEXT };
      if (FINAL_LANGUAGE) payload.language = FINAL_LANGUAGE;
      browserWs.send(JSON.stringify(payload));
      console.log("[mock-stt-bridge] sent transcript.final:", payload);
    }, 400);
  });

  browserWs.on("close", () => console.log("[mock-stt-bridge] browser disconnected"));
});
