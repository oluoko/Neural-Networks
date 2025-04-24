export interface ClassificationResult {
  genre: string;
  confidence: number;
  genreConfidences: {
    genre: string;
    confidence: number;
  }[];
}

export interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

export interface AudioPlayerProps {
  audioSrc: string | null;
}

export interface ClassificationResultProps {
  result: ClassificationResult | null;
  audioSrc: string | null;
}
