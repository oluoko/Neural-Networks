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
    <main className="min-h-screen bg-gray-50 py-6 sm:py-12 px-4">
      <div className="max-w-[95vw] sm:max-w-[90vw] mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-8">
          <div className="md:col-span-2 text-center md:text-left mb-6 md:mb-0">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-2">
              Music Genre Classifier
            </h1>
            <p className="text-lg sm:text-xl text-gray-600 mb-6">
              Upload an audio file to identify its music genre using machine
              learning
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center md:justify-start gap-4 sm:gap-6">
              <FileUpload
                onFileSelect={handleFileSelect}
                isLoading={isLoading}
              />
              {isLoading && <LoadingSpinner />}
            </div>

            <div className="mt-6">
              <AudioPlayer audioSrc={audioSrc} />
            </div>
          </div>

          <div className="md:col-span-1">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 md:hidden">
              Classification Results
            </h2>
            <ClassificationResult result={result} audioSrc={audioSrc} />
          </div>
        </div>
      </div>
    </main>
  );
}
