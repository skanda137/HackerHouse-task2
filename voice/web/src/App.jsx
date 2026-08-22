import { useRef, useState } from "react";
import "./App.css";

function App() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [status, setStatus] = useState("Ready");

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  const startRecording = async () => {
    try {
      setStatus("Connecting...");
      
      const ws = new WebSocket("ws://localhost:8080");

      ws.onopen = async () => {
        setStatus("Listening...");
        setRecording(true);

        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });

        const recorder = new MediaRecorder(stream, {
          mimeType: "audio/webm",
        });

        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = async (event) => {
          if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            const buffer = await event.data.arrayBuffer();
            ws.send(buffer);
          }
        };

        recorder.start(250);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.transcript) {
            setTranscript(data.transcript);
          }
        } catch {
          console.log("Server:", event.data);
        }
      };

      ws.onerror = () => {
        setStatus("Connection error");
        setRecording(false);
      };

      ws.onclose = () => {
        setRecording(false);
        setStatus("Disconnected");
      };

      wsRef.current = ws;
    } catch (error) {
      console.error(error);
      setStatus("Microphone error");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();

      const tracks =
        mediaRecorderRef.current.stream?.getTracks() || [];

      tracks.forEach((track) => track.stop());
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    setRecording(false);
    setStatus("Ready");
  };

  return (
    <div className="app">
      <h1>Voice RAG</h1>

      <p className="status">{status}</p>

      <button
        className={recording ? "recording" : ""}
        onClick={recording ? stopRecording : startRecording}
      >
        {recording ? "Stop Recording" : "🎙️ Start Speaking"}
      </button>

      <div className="transcript">
        <h2>Transcript</h2>
        <p>{transcript || "Your speech will appear here..."}</p>
      </div>
    </div>
  );
}

export default App;