import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const RAG_API_BASE = "http://localhost:8000";
const STT_WS_URL = "ws://localhost:8080";
// server.js opens the Sarvam session with language_code=auto and forwards
// the detected `language` field on transcript.final; this is only the
// fallback when that field is missing or unrecognized.
const QUERY_LANGUAGE = "en";

const TTS_LANG_MAP = { hi: "hi-IN", ta: "ta-IN", te: "te-IN", bn: "bn-IN", en: "en-US" };
const KNOWN_LANGUAGES = new Set(["hi", "ta", "te", "bn", "en"]);

// Sarvam's realtime API only reports `language` on the transcript event when
// the session is opened with language_code=auto (BCP-47 codes like "hi-IN").
// server.js pins the session to en-IN, so this normally falls back to
// QUERY_LANGUAGE -- kept so the frontend is correct if that's ever changed.
function normalizeLanguage(rawLanguage) {
  if (!rawLanguage) return null;
  const code = rawLanguage.split("-")[0].toLowerCase();
  return KNOWN_LANGUAGES.has(code) ? code : null;
}

function speak(text, lang) {
  if (!text || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = TTS_LANG_MAP[lang] || "en-US";
  window.speechSynthesis.speak(utterance);
}

function App() {
  // idle | listening | thinking | answer | guardrail | error
  const [phase, setPhase] = useState("idle");
  const [statusText, setStatusText] = useState("Ready");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const wsRef = useRef(null);
  const audioRef = useRef(null);
  const recordStartRef = useRef(0);
  const submittedRef = useRef(false);
  const phaseRef = useRef(phase);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  const stopMic = useCallback(() => {
    if (audioRef.current) {
      const { stream, audioContext, source, workletNode } = audioRef.current;
      workletNode.port.onmessage = null;
      workletNode.disconnect();
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      audioContext.close();
      audioRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const askBackend = useCallback(async (query, sttLatencyMs, language) => {
    setPhase("thinking");
    setStatusText("Thinking...");
    setAnswer("");
    setConfidence(null);
    setErrorMessage("");

    try {
      const res = await fetch(`${RAG_API_BASE}/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          query_language: language,
          stt_latency_ms: sttLatencyMs,
        }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Backend returned ${res.status}: ${body}`);
      }

      const data = await res.json();

      if (data.status === "success") {
        setAnswer(data.answer);
        setConfidence(data.confidence_score);
        setPhase("answer");
        setStatusText("Answered");
        speak(data.answer, language);
      } else if (data.status === "fallback_no_context" || data.status === "fallback_ungrounded") {
        // Guardrail declining to answer -- a feature, not a failure.
        setAnswer(data.answer);
        setConfidence(data.confidence_score);
        setPhase("guardrail");
        setStatusText("Declined to answer");
        speak(data.answer, language);
      } else {
        // status === "error": the harness itself failed (e.g. generation timeout).
        setErrorMessage(data.answer || "The RAG service failed to generate a response.");
        setPhase("error");
        setStatusText("Generation failed");
      }
    } catch (err) {
      console.error("RAG backend call failed:", err);
      setErrorMessage(err.message || "Could not reach the RAG backend.");
      setPhase("error");
      setStatusText("Connection error");
    }
  }, []);

  const startRecording = async () => {
    setPhase("listening");
    setStatusText("Connecting...");
    setTranscript("");
    setAnswer("");
    setErrorMessage("");
    submittedRef.current = false;

    try {
      const ws = new WebSocket(STT_WS_URL);
      wsRef.current = ws;

      ws.onopen = async () => {
        setStatusText("Listening...");
        recordStartRef.current = performance.now();

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 16000 },
          });

          // Sarvam's realtime STT expects raw linear16 (16-bit PCM) mono
          // audio at 16kHz on the binary channel, not a container format.
          // Requesting the AudioContext at 16kHz makes the browser resample
          // the mic input for us; the worklet then converts Float32 -> Int16.
          const audioContext = new AudioContext({ sampleRate: 16000 });
          await audioContext.audioWorklet.addModule("/pcm-worklet.js");

          const source = audioContext.createMediaStreamSource(stream);
          const workletNode = new AudioWorkletNode(audioContext, "pcm-worklet-processor");

          workletNode.port.onmessage = (event) => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(event.data);
            }
          };

          source.connect(workletNode);
          workletNode.connect(audioContext.destination);

          audioRef.current = { stream, audioContext, source, workletNode };
        } catch (micErr) {
          console.error(micErr);
          setErrorMessage("Microphone access failed.");
          setPhase("error");
          setStatusText("Microphone error");
          ws.close();
        }
      };

      ws.onmessage = (event) => {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          console.log("Non-JSON message from STT bridge:", event.data);
          return;
        }

        if (data.event === "transcript.partial") {
          setTranscript(data.text || "");
          return;
        }

        if (data.event === "error") {
          console.error("STT bridge reported an error:", data.message);
          if (!submittedRef.current) {
            setErrorMessage(data.message || "The speech-to-text service failed.");
            setPhase("error");
            setStatusText("Speech-to-text error");
            stopMic();
          }
          return;
        }

        if (data.event === "transcript.final") {
          const finalText = (data.text || "").trim();
          const language = normalizeLanguage(data.language) || QUERY_LANGUAGE;
          setTranscript(finalText);

          if (finalText && !submittedRef.current) {
            submittedRef.current = true;
            const sttLatencyMs = performance.now() - recordStartRef.current;
            stopMic();
            askBackend(finalText, sttLatencyMs, language);
          }
          return;
        }

        // Defensive fallback in case the bridge is ever pointed at the
        // legacy Sarvam schema ({type:"data", data:{transcript}}) instead
        // of the realtime one this app targets.
        if (data.transcript) {
          setTranscript(data.transcript);
        }
      };

      ws.onerror = () => {
        console.error("STT WebSocket error");
        if (!submittedRef.current) {
          setErrorMessage("Lost connection to the speech-to-text bridge.");
          setPhase("error");
          setStatusText("Connection error");
          stopMic();
        }
      };

      ws.onclose = () => {
        if (!submittedRef.current && phaseRef.current === "listening") {
          setPhase("idle");
          setStatusText("Ready");
        }
      };
    } catch (error) {
      console.error(error);
      setErrorMessage("Could not open the speech-to-text connection.");
      setPhase("error");
      setStatusText("Connection error");
    }
  };

  const stopRecording = () => {
    stopMic();
    setPhase("idle");
    setStatusText("Ready");
  };

  const recording = phase === "listening";

  return (
    <div className="app">
      <h1>Voice RAG</h1>

      <p className={`status status-${phase}`}>{statusText}</p>

      <button
        className={recording ? "mic-btn recording" : "mic-btn"}
        onClick={recording ? stopRecording : startRecording}
        disabled={phase === "thinking"}
      >
        {recording ? "Stop Recording" : "🎙️ Start Speaking"}
      </button>

      <div className="transcript-box">
        <h2>Transcript</h2>
        <p>{transcript || "Your speech will appear here..."}</p>
      </div>

      {phase === "thinking" && (
        <div className="state-card thinking">
          <span className="spinner" aria-hidden="true" />
          <p>Retrieving context and generating an answer…</p>
        </div>
      )}

      {phase === "answer" && (
        <div className="state-card answer">
          <h2>Answer</h2>
          <p>{answer}</p>
          {confidence !== null && (
            <p className="meta">grounding confidence: {(confidence * 100).toFixed(0)}%</p>
          )}
        </div>
      )}

      {phase === "guardrail" && (
        <div className="state-card guardrail">
          <h2>⚠ No grounded answer</h2>
          <p>{answer || "I don't have grounded information to answer that."}</p>
          <p className="meta">
            The retrieval system found no confident, grounded match for this question in the
            dataset, so it's declining to answer rather than guessing.
          </p>
        </div>
      )}

      {phase === "error" && (
        <div className="state-card error">
          <h2>Something went wrong</h2>
          <p>{errorMessage}</p>
        </div>
      )}
    </div>
  );
}

export default App;
