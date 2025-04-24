"use client";

import { useState } from "react";
import FileUpload from "@/components/FileUpload";
import AudioPlayer from "@/components/AudioPlayer";
import LoadingSpinner from "@/components/LoadingSpinner";
import { ClassificationResult as ClassificationResultType } from "@/lib/types";
import ClassificationResult from "@/components/ClassificationResults";

export default function Home() {
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResultType | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileSelect = async (file: File) => {
    // Reset state
    setResult(null);
    setIsLoading(true);

    // Create audio URL for preview
    const audioUrl = URL.createObjectURL(file);
    setAudioSrc(audioUrl);

    // Create form data to send to the API
    const formData = new FormData();
    formData.append("audio", file);

    try {
      // Send the audio file to the API
      const response = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to classify audio");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error classifying audio:", error);
      alert("Error classifying audio. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 py-12 px-4">
      <div className="max-w-3xl mx-auto">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Music Genre Classifier
          </h1>
          <p className="text-xl text-gray-600">
            Upload an audio file to identify its music genre using machine
            learning
          </p>
        </div>

        <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />

        {isLoading && <LoadingSpinner />}

        <AudioPlayer audioSrc={audioSrc} />

        <ClassificationResult result={result} audioSrc={audioSrc} />
      </div>
    </main>
  );
}
