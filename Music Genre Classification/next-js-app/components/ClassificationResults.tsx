import { ClassificationResultProps } from "@/lib/types";

// Map numeric genre IDs to human-readable genre names
const genreMapping: Record<string, string> = {
  "0": "Classical",
  "1": "Country",
  "2": "Electronic/Dance",
  "3": "Hip-Hop/Rap",
  "4": "Jazz",
  "5": "Metal",
  "6": "Pop",
  "7": "R&B/Soul",
  "8": "Rock",
  "9": "Folk/Acoustic",
};

export default function ClassificationResult({
  result,
  audioSrc,
}: ClassificationResultProps) {
  if (!result || !audioSrc) return null;

  // Get the top genre confidence for the progress bar
  const topConfidence = result.confidence;

  // Map the genre ID to a human-readable name
  const getGenreName = (genreId: string) => {
    return genreMapping[genreId] || `Genre ${genreId}`;
  };

  // Generate a color based on the confidence level
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-500";
    if (confidence >= 0.5) return "bg-yellow-500";
    return "bg-red-500";
  };

  const getConfidenceEmoji = (confidence: number) => {
    if (confidence >= 0.8) return "🎯";
    if (confidence >= 0.5) return "👍";
    return "🤔";
  };

  return (
    <div className="w-full max-w-md mx-auto mt-8 bg-white p-6 rounded-lg shadow-md text-black/80">
      <h2 className="text-2xl font-bold mb-4 text-center">
        Classification Result
      </h2>
      <div className="mb-6 text-center">
        <p className="text-3xl font-bold mb-2">{getGenreName(result.genre)}</p>
        <div className="flex items-center justify-center">
          <span className="text-xl mr-2">
            {getConfidenceEmoji(topConfidence)}
          </span>
          <span className="text-lg font-medium">
            {Math.round(topConfidence * 100)}% confidence
          </span>
        </div>
      </div>
      <div className="space-y-4">
        <h3 className="text-lg font-medium">All Genre Probabilities:</h3>
        {result.genreConfidences
          .sort((a, b) => b.confidence - a.confidence)
          .map((item, index) => (
            <div key={index} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>{getGenreName(item.genre)}</span>
                <span>{Math.round(item.confidence * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`${getConfidenceColor(
                    item.confidence
                  )} h-2 rounded-full`}
                  style={{ width: `${item.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
