import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import { promises as fs } from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";
import os from "os";

export async function POST(request: NextRequest) {
  try {
    // Check if the request is multipart form data
    if (!request.headers.get("content-type")?.includes("multipart/form-data")) {
      return NextResponse.json(
        { error: "Request must be multipart/form-data" },
        { status: 400 }
      );
    }

    // Parse form data
    const formData = await request.formData();
    const audioFile = formData.get("audio") as File;

    if (!audioFile) {
      return NextResponse.json(
        { error: "No audio file provided" },
        { status: 400 }
      );
    }

    // Create a temporary directory for processing
    const tempDir = path.join(os.tmpdir(), uuidv4());
    await fs.mkdir(tempDir, { recursive: true });

    // Save the audio file locally for processing
    const audioPath = path.join(tempDir, audioFile.name);
    const audioBuffer = Buffer.from(await audioFile.arrayBuffer());
    await fs.writeFile(audioPath, audioBuffer);

    // Get the absolute path to the python script
    const scriptPath = path.join(process.cwd(), "python", "inference.py");
    const modelDir = path.join(process.cwd(), "public", "model");

    // Run the Python script for classification
    const result = await new Promise<string>((resolve, reject) => {
      exec(
        `python "${scriptPath}" "${audioPath}" "${modelDir}"`,
        (error, stdout, stderr) => {
          if (error) {
            console.error(`Execution error: ${error}`);
            console.error(`Stderr: ${stderr}`);
            reject(error);
            return;
          }
          resolve(stdout);
        }
      );
    });

    // Clean up temporary files
    try {
      await fs.rm(tempDir, { recursive: true });
    } catch (e) {
      console.error("Error cleaning up temp files:", e);
    }

    // Parse the Python script output
    const classificationResult = JSON.parse(result);

    return NextResponse.json(classificationResult);
  } catch (error) {
    console.error("Error processing audio:", error);
    return NextResponse.json(
      { error: "Failed to process audio file" },
      { status: 500 }
    );
  }
}
