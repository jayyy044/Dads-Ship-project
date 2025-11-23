from google import genai
import logging
import os
import time
import tempfile
import re

class TextExtractor:
    
    def __init__(self, modelName: str):
        self.logger = logging.getLogger(__name__)
        apiKey = os.getenv("GEMINI_KEY")
        self.geminiClient = genai.Client(api_key=apiKey)
        self.modelName = modelName
        self.generationConfig = {
            "temperature": 0.0,
            "max_output_tokens": 8192,
        }
        self.extractionPrompt = """
            You are a resume text processing assistant. Your task is to extract and structure text from resume PDFs while maintaining the original content and organization exactly as presented.
            Your Task
            Transform the PDF resume into clean, well-structured text while following these strict rules:
            1. Section Preservation

            Identify all sections present in the resume (e.g., Education, Experience, Projects, Skills, Summary, Certifications, etc.)
            CRITICAL: Preserve sections in the EXACT order they appear in the original document
            Do NOT reorder sections based on convention - maintain the author's chosen sequence
            Keep section headers as they appear (maintain capitalization and formatting style)

            2. Content Integrity Rules

            Never modify, paraphrase, or rewrite any content - preserve text character-for-character
            Keep all dates, names, titles, and details exactly as written
            Maintain all technical terms, acronyms, and proper nouns without changes
            Preserve the original tone, word choice, and phrasing
            If information appears in a particular section in the original, keep it in that same section

            3. Formatting Guidelines

            Use consistent spacing: one blank line between sections
            Preserve bullet points using a consistent symbol (•, -, or * as they appear)
            Maintain hierarchical structure (headers, subheaders, entries, bullet points)
            Remove artifacts like page numbers, headers/footers, or PDF metadata
            Fix obvious OCR/extraction errors (e.g., garbled characters, "Exper1ence" → "Experience") but do NOT change actual content
            Preserve any formatting emphasis like ALL CAPS section headers if present in the original

            4. Structure Handling

            Contact information should appear at the top
            Each section should be clearly delineated with its header
            Maintain the spacing and grouping of related information
            Preserve date alignments and formatting patterns
            Keep any subsections in their original positions

            5. Quality Requirements
            Before outputting, verify:

            All sections appear in their original order
            No content has been modified, added, or removed
            All bullet points are verbatim from the original
            Section headers are preserved as they appeared
            Formatting is clean and readable
            All information remains in its original section

            Output Format
            Return only the cleaned, structured resume text with no explanatory comments, metadata, or surrounding markdown code blocks. The output should be plain text, ready to use as-is.
            Important Notes

            Do not make assumptions about what "should" be in each section
            Do not reorganize content to fit standard resume conventions
            Focus on faithful extraction and clean formatting
            Preserve the resume author's organizational choices

            Extract and structure the resume from the uploaded PDF file.
        """
       
        self.logger.info("Text extractor Initialized")

    def processingResume(self, uploadedFile) -> None:
        """Wait for file processing to complete."""
        while hasattr(uploadedFile, 'state') and uploadedFile.state.name == "PROCESSING":
            self.logger.info("File processing...")
            time.sleep(2)
            uploadedFile = self.geminiClient.files.get(name=uploadedFile.name)
        
        if hasattr(uploadedFile, 'state') and uploadedFile.state.name == "FAILED":
            msg = "File processing failed"
            self.logger.error(msg)
            raise Exception(msg)

    def delProcessedResume(self, uploadedFile) -> None:
        """Safely delete uploaded file."""
        if uploadedFile and hasattr(uploadedFile, 'name'):
            try:
                self.geminiClient.files.delete(name=uploadedFile.name)
                self.logger.info("Temporary file deleted successfully")
            except Exception as e:
                self.logger.warning(f"Failed to delete temporary file: {e}")

    async def extractFromPdf(self, resumePdf: str) -> str:
        """Extract text from PDF file."""
        uploadedFile = None
        tempFilePath = None
        try:
            fileContent = await resumePdf.read()
            fileSize = len(fileContent)
            filename = resumePdf.filename

            if fileSize == 0:
                raise ValueError(f"Resume file '{filename}' is empty (0 bytes)")
            
            self.logger.info(f"Processing uploaded file: {filename} ({fileSize / 1024:.2f}KB)")
            
            # Write to temporary file so Gemini can process it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpFile:
                tmpFile.write(fileContent)
                tempFilePath = tmpFile.name
            
            uploadPath = tempFilePath
            uploadedFile = self.geminiClient.files.upload(
                file=uploadPath, 
                config={"mimeType": "application/pdf"}
            )
            self.logger.info(f"File uploaded: {uploadedFile.name}")
            
            self.processingResume(uploadedFile)
            
            # Extract text
            response = self.geminiClient.models.generate_content(
                model=self.modelName,
                contents=[
                    self.extractionPrompt,
                    uploadedFile
                ],
                config=self.generationConfig
            )

            text = getattr(response, "text", str(response))
            self.logger.info(f"Extraction completed. Length: {len(text)} chars")
            return text
            
        except ValueError as ve:
            # Log and re-raise validation errors
            self.logger.error(f"Validation error for file '{resumePdf.filename}': {ve}")
            raise
            
        except Exception as e:
            # Log the full exception with traceback
            self.logger.error(
                f"Extraction failed for file '{resumePdf.filename}': {type(e).__name__}: {e}",
                exc_info=True
            )
            raise Exception(f"Failed to extract text from resume: {e}") from e
        
        finally:
            self.delProcessedResume(uploadedFile)
            if tempFilePath and os.path.exists(tempFilePath):
                try:
                    os.unlink(tempFilePath)
                    self.logger.info("Temporary file cleaned up")
                except Exception as e:
                    self.logger.warning(f"Failed to clean temporary file: {e}")
                    raise

        
       