'use client';

import { useState } from 'react';

// List of instruments that matches the backend
const VALID_INSTRUMENTS = [
  "lead", "pluck", "keys", "pad", "drums", 
  "kick", "snare", "closed_hat", "open_hat", "clap"
];

// URL
const API_BASE_URL = "http://127.0.0.1:8000";

export default function HomePage() {
  const [selectedInstrument, setSelectedInstrument] = useState<string>('lead');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateMidi = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // 1. Construct the request URL with the selected instrument as a query parameter
      const response = await fetch(
        `${API_BASE_URL}/generate-midi?instrument=${selectedInstrument}`
      );

      if (!response.ok) {
        // If the server returns an error, try to parse it as JSON
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! Status: ${response.status}`);
      }

      // 2. The response body is the MIDI file itself. We get it as a Blob.
      const midiBlob = await response.blob();

      // 3. Create a temporary URL for the Blob
      const url = window.URL.createObjectURL(midiBlob);

      // 4. Create a temporary link element to trigger the download
      const a = document.createElement('a');
      a.href = url;
      a.download = `${selectedInstrument}_pattern.mid`; // The filename for the download
      document.body.appendChild(a);
      a.click();

      // 5. Clean up by removing the link and revoking the temporary URL
      a.remove();
      window.URL.revokeObjectURL(url);

    } catch (err: any) {
      console.error("Failed to generate MIDI:", err);
      setError(err.message || "An unknown error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gray-900 text-white">
      <div className="flex flex-col items-center gap-6 p-10 rounded-lg bg-gray-800 shadow-xl">
        <h1 className="text-4xl font-bold">MidiGEN</h1>
        <p className="text-gray-400">Select an instrument and generate a pattern.</p>
        
        <div className="flex items-center gap-4">
          <label htmlFor="instrument-select" className="font-medium">Instrument:</label>
          <select
            id="instrument-select"
            value={selectedInstrument}
            onChange={(e) => setSelectedInstrument(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:outline-none"
          >
            {VALID_INSTRUMENTS.map((inst) => (
              <option key={inst} value={inst}>
                {inst.charAt(0).toUpperCase() + inst.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handleGenerateMidi}
          disabled={isLoading}
          className="w-full px-6 py-3 bg-blue-600 rounded-md text-lg font-semibold hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed transition-all duration-200"
        >
          {isLoading ? 'Generating...' : 'Generate MIDI'}
        </button>

        {error && <p className="text-red-400 mt-4">Error: {error}</p>}
      </div>
    </main>
  );
}