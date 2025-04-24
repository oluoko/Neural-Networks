import { useEffect, useRef } from "react";
import { AudioPlayerProps } from "../lib/types";

export default function AudioPlayer({ audioSrc }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (audioRef.current && audioSrc) {
      audioRef.current.load();
    }
  }, [audioSrc]);

  if (!audioSrc) return null;

  return (
    <div className="w-full max-w-md mx-auto mt-8 bg-gray-100 p-4 rounded-lg">
      <h3 className="text-lg font-medium mb-2">Audio Preview</h3>
      <audio ref={audioRef} controls className="w-full">
        <source src={audioSrc} type="audio/mpeg" />
        Your browser does not support the audio element.
      </audio>
    </div>
  );
}
