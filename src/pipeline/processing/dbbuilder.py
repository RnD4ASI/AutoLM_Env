import os
from typing import List, Any, Dict, Optional, Union, Tuple
import subprocess
import spacy
from spacy.cli import download
import pandas as pd
import networkx as nx
import re
from markitdown import MarkItDown
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import json
import networkx as nx
import glob
import time
import traceback
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from src.pipeline.shared.logging import get_logger
from src.pipeline.processing.generator import Generator, MetaGenerator
from src.pipeline.shared.utility import DataUtility


logger = get_logger(__name__)

class TextParser:
    """
    Provides methods for cleansing data including PDF to Markdown conversion and text chunking.
    """
    def __init__(self) -> None:
        """Initialize a TextParser instance.
        No parameters required.
        Returns:
            None
        """
        # Initialize any resources, e.g., spaCy model if needed
        logger.debug("TextParser initialization started")
        try:
            start_time = time.time()
            try:
                # Attempt to load the small English model
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                # Model not found, download then load
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            logger.debug(f"Loaded spaCy model in {time.time() - start_time:.2f} seconds")

            # Initialize the Generator
            self.generator = Generator()
            logger.debug("TextParser initialized successfully")
        except Exception as e:
            logger.error(f"TextParser initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise

    def pdf2md_markitdown(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using the MarkItDown package.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If MarkItDown package is not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_markitdown (MarkItDown) for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize converter and convert PDF to Markdown
            start_time = time.time()
            logger.debug(f"Initializing MarkItDown converter")
            converter = MarkItDown()
            logger.debug(f"Starting PDF conversion: {pdf_path} -> {md_path}")
            converter.convert_file(pdf_path, md_path)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")

    def pdf2md_openleaf(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using the openleaf-markdown-pdf shell command.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            RuntimeError: If the openleaf-markdown-pdf command fails or isn't installed.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_openleaf (OpenLeaf) for {pdf_path}")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'

        try:
            # Run the openleaf-markdown-pdf command
            cmd = ['openleaf-markdown-pdf', '--input', pdf_path, '--output', md_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully converted PDF to Markdown: {md_path}")
                return None
            else:
                error_msg = result.stderr or "Unknown error occurred"
                logger.error(f"Command failed: {error_msg}")
                raise RuntimeError(f"PDF conversion failed: {error_msg}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to execute openleaf-markdown-pdf: {str(e)}")
            raise RuntimeError(f"openleaf-markdown-pdf command failed: {str(e)}")
        except FileNotFoundError:
            logger.error("openleaf-markdown-pdf command not found. Please install it first.")
            raise RuntimeError("openleaf-markdown-pdf is not installed")

    # def pdf2md_ocr(self, pdf_path: str, md_path: str, model: str = "GOT-OCR2") -> None:
    #     """Convert a PDF to Markdown using an open sourced model from HuggingFace.
    #     tmp/ocr folder is used to store the temporary images.

    #     Parameters:
    #         pdf_path (str): The file path to the PDF.
    #         md_path (str): The file path to save the generated Markdown.
    #         model (str): The model to use for conversion (default is "GOT-OCR2").

    #     Returns:
    #         None

    #     Raises:
    #         FileNotFoundError: If the PDF file doesn't exist.
    #         ImportError: If required packages are not installed.
    #         RuntimeError: If conversion fails.
    #     """
    #     # Convert PDF pages to images
    #     logger.info(f"Converting PDF to Markdown using pdf2md_ocr (OCR Model) for {pdf_path}")
    #     if pdf_path is None or not os.path.exists(pdf_path):
    #         raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
    #     pages = convert_from_path(pdf_path)
    #     if not pages:
    #         raise ValueError("No pages found in the PDF.")
        
    #     tmp_dir = Path.cwd() / "tmp/ocr"
    #     tmp_dir.mkdir(exist_ok=True)
        
    #     # Save each image temporarily and collect their file paths
    #     try:
    #         image_paths = []
    #         for idx, page in enumerate(pages):
    #             image_path = tmp_dir / f"temp_page_{idx}.jpg"
    #             page.save(image_path, "JPEG")
    #             image_paths.append(image_path)
            
    #         # Execute OCR on all temporary image files
    #         ocr_text = self.generator.get_ocr(image_paths=image_paths, model=model)
    #         with open(md_path, "w", encoding="utf-8") as f:
    #             f.write(ocr_text)
            
    #         # Clean up temporary files
    #         for image_path in image_paths:
    #             os.remove(image_path)
    #         logger.info(f"PDF to Markdown conversion completed.")

    #     except Exception as e:
    #         logger.error(f"PDF to Markdown conversion failed: {str(e)}")
    #         raise e

    def pdf2md_llamaindex(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using LlamaIndex and PyMuPDF.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If required packages are not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_llamaindex for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        try:
            # Import required packages
            import pymupdf4llm
            from llama_index.core import Document
        except ImportError:
            logger.error("Required packages not installed. Please install using: pip install pymupdf4llm llama-index")
            raise ImportError("Required packages not installed: pymupdf4llm, llama-index")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize LlamaIndex processor options
            llamaindex_options = {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'use_chunks': False
            }
            
            # Extract text using LlamaIndex and PyMuPDF
            start_time = time.time()
            logger.debug(f"Initializing LlamaIndexPDFProcessor with options: {llamaindex_options}")
            
            # Create a LlamaMarkdownReader with appropriate options
            import inspect
            reader_params = inspect.signature(pymupdf4llm.LlamaMarkdownReader.__init__).parameters
            
            # Prepare kwargs based on available parameters
            kwargs = {}
            
            # Add options if they are supported by the current version
            if 'chunk_size' in reader_params:
                kwargs['chunk_size'] = llamaindex_options['chunk_size']
                
            if 'chunk_overlap' in reader_params:
                kwargs['chunk_overlap'] = llamaindex_options['chunk_overlap']
            
            # Create reader with configured options
            logger.debug(f"Creating LlamaMarkdownReader with parameters: {kwargs}")
            llama_reader = pymupdf4llm.LlamaMarkdownReader(**kwargs)
            
            # Load and convert the PDF to LlamaIndex documents
            load_data_params = inspect.signature(llama_reader.load_data).parameters
            load_kwargs = {}
            
            # Add any additional load_data parameters if supported
            if 'use_chunks' in load_data_params:
                load_kwargs['use_chunks'] = llamaindex_options['use_chunks']
                
            logger.debug(f"Loading PDF with parameters: {load_kwargs}")
            documents = llama_reader.load_data(str(pdf_path), **load_kwargs)
            
            # Combine all documents into a single markdown text
            if llamaindex_options['use_chunks']:
                # Return documents as they are (already chunked by LlamaIndex)
                markdown_text = "\n\n---\n\n".join([doc.text for doc in documents])
            else:
                # Combine all text into a single document
                markdown_text = "\n\n".join([doc.text for doc in documents])
            
            # Write the markdown output to file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown using LlamaIndex: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")
    
    def pdf2md_pymupdf(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using PyMuPDF directly.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If pymupdf4llm package is not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_pymupdf for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        try:
            # Import required packages
            import pymupdf4llm
            import inspect
        except ImportError:
            logger.error("pymupdf4llm package not installed. Please install using: pip install pymupdf4llm")
            raise ImportError("pymupdf4llm package not installed")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize PyMuPDF processor options
            pymupdf_options = {
                'preserve_images': False,
                'preserve_tables': True
            }
            
            start_time = time.time()
            logger.debug(f"Using PyMuPDF with options: {pymupdf_options}")
            
            # Use pymupdf4llm to convert directly to markdown
            # Check pymupdf4llm version to see if it supports the options
            to_markdown_params = inspect.signature(pymupdf4llm.to_markdown).parameters
            
            # Prepare kwargs based on available parameters
            kwargs = {}
            
            # Add options if they are supported by the current version
            if 'preserve_images' in to_markdown_params:
                kwargs['preserve_images'] = pymupdf_options['preserve_images']
                
            if 'preserve_tables' in to_markdown_params:
                kwargs['preserve_tables'] = pymupdf_options['preserve_tables']
                
            # Call to_markdown with appropriate options
            logger.debug(f"Converting PDF to Markdown with parameters: {kwargs}")
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path), **kwargs)
            
            # Write the markdown output to file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown using PyMuPDF: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")
    
    def pdf2md_textract(self, pdf_path: str) -> None:
        """Convert a PDF to Markdown using the textract library.

        Parameters:
            pdf_path (str): The file path to the PDF.

        Returns:
            None

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If textract package is not installed.
            RuntimeError: If conversion fails.
        """
        logger.info(f"Converting PDF to Markdown using pdf2md_textract for {pdf_path}")
        logger.debug(f"PDF path details: {os.path.abspath(pdf_path)}, size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")
        
        try:
            import textract
        except ImportError:
            logger.error("textract package not installed. Please install using: pip install textract")
            raise ImportError("textract package not installed")

        # Verify PDF file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create output path for markdown file
        md_path = os.path.splitext(pdf_path)[0] + '.md'
        
        try:
            # Initialize textract options
            textract_options = {
                'method': 'pdftotext',
                'encoding': 'utf-8',
                'layout': True
            }
            
            start_time = time.time()
            logger.debug(f"Using textract with options: {textract_options}")
            
            # Build the extraction options
            extract_kwargs = {
                'method': textract_options['method'],
                'encoding': textract_options['encoding'],
                'layout': textract_options['layout']
            }
            
            # Remove None values
            extract_kwargs = {k: v for k, v in extract_kwargs.items() if v is not None}
            
            # Extract text from PDF
            logger.debug(f"Extracting text with parameters: {extract_kwargs}")
            text = textract.process(str(pdf_path), **extract_kwargs).decode(textract_options['encoding'])
            
            # Convert plain text to basic markdown
            # This is a simple conversion since textract doesn't preserve formatting well
            lines = text.split('\n')
            markdown_lines = []
            in_paragraph = False
            
            for line in lines:
                line = line.strip()
                if not line:  # Empty line
                    if in_paragraph:
                        markdown_lines.append('')  # End paragraph
                        in_paragraph = False
                else:
                    # Very basic heuristic for headings: all caps, not too long
                    if line.isupper() and len(line) < 100:
                        markdown_lines.append(f"## {line}")
                        in_paragraph = False
                    else:
                        if not in_paragraph:
                            in_paragraph = True
                        markdown_lines.append(line)
            
            markdown_text = '\n'.join(markdown_lines)
            
            # Write the markdown output to file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            conversion_time = time.time() - start_time
            logger.info(f"Successfully converted PDF to Markdown: {md_path} in {conversion_time:.2f} seconds")
            if os.path.exists(md_path):
                logger.debug(f"Generated markdown file size: {os.path.getsize(md_path)} bytes")
            return None
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to Markdown using textract: {str(e)}")
            logger.debug(f"PDF conversion error details: {traceback.format_exc()}")
            raise RuntimeError(f"PDF conversion failed: {str(e)}")


class TextChunker:
    """Handles text chunking and markdown processing operations.
    
    This class provides methods for splitting markdown documents into chunks using different strategies:
    1. Length-based chunking: Divides text into fixed-size chunks with configurable overlap
    2. Hierarchy-based chunking: Divides text based on heading structure and document hierarchy
    
    The chunking parameters (chunk size and overlap) can be configured through the main_config.json file
    or passed directly to the chunking methods. Default values are used as fallbacks.
    
    Attributes:
        nlp: spaCy language model for text processing
        default_chunk_size: Default size of chunks in characters (from config or 1000)
        default_chunk_overlap: Default overlap between chunks in characters (from config or 100)
    """
    
    def __init__(self, config_file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize TextChunker with default configuration from main_config.json.
        
        Parameters:
            config_file_path (Optional[Union[str, Path]]): Path to the configuration file.
                If None, defaults to Path.cwd() / "config" / "main_config.json".
                
        Returns:
            None
        """
        logger.debug("TextChunker initialization started")
        try:
            start_time = time.time()
            # Load spaCy model
            
            self.datautility = DataUtility()
            
            # Load defaults from config file
            chunker_config_path = Path(config_file_path) if config_file_path else Path.cwd() / "config" / "main_config.json"
            try:
                if os.path.exists(chunker_config_path):
                    with open(chunker_config_path, 'r') as f:
                        config = json.load(f)
                    # Get chunking configuration from main_config.json
                    chunker_config = config.get('knowledge_base', {}).get('chunking', {}).get('other_config', {})
                    self.default_chunk_size = chunker_config.get('chunk_size', 5000)
                    self.default_chunk_overlap = chunker_config.get('chunk_overlap', 50)
                    logger.debug(f"Loaded chunking configuration from {chunker_config_path}: chunk_size={self.default_chunk_size}, chunk_overlap={self.default_chunk_overlap}")
                else:
                    logger.warning(f"Config file not found at {chunker_config_path}, using default values")
                    self.default_chunk_size = 5000
                    self.default_chunk_overlap = 50
                    
            except Exception as config_error:
                logger.error(f"Error loading config: {str(config_error)}, using default values")
                self.default_chunk_size = 5000
                self.default_chunk_overlap = 50
                
            build_time = time.time() - start_time
            logger.debug(f"Loaded spaCy model in {build_time:.2f} seconds")
            logger.debug("TextChunker initialized successfully")
        except Exception as e:
            logger.error(f"TextChunker initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise

    def length_based_chunking(self, markdown_file: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> pd.DataFrame:
        """Chunks text from a markdown file into overlapping segments and returns as DataFrame.
        This is a pure length-based chunking method that doesn't consider headings.

        Parameters:
            markdown_file (str): Path to the markdown file.
            chunk_size (int, optional): Size of each chunk in characters. If None, uses the default_chunk_size.
            overlap (int, optional): Number of characters to overlap between chunks. If None, uses the default_chunk_overlap.

        Returns:
            pd.DataFrame: DataFrame containing:
                - source: Source of the document
                - document_id: Unique identifier for source document
                - chunk_id: Unique identifier for text chunk
                - document_name: Name of the source document
                - reference: Unique identifier for each chunk
                - hierarchy: Document name (no heading hierarchy)
                - corpus: The chunk text content
                - embedding_model: Model used to generate embeddings (None at this stage)

        Raises:
            FileNotFoundError: If markdown file doesn't exist
            ValueError: If chunk_size <= overlap
        """
        # Use default values if not provided
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        if overlap is None:
            overlap = self.default_chunk_overlap
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
            
        # Get filename for reference generation
        filename = Path(markdown_file).stem.upper()
        logger.debug(f"Processing {filename} with chunk_size={chunk_size}, overlap={overlap}")
            
        # Read markdown file
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            logger.error(f"Markdown file not found: {markdown_file}")
            raise
            
        # Initialize chunks list
        chunks = []
        start = 0
        chunk_id = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate chunk boundaries
            end = min(start + chunk_size, text_length)
            
            # Get chunk text
            chunk_text = text[start:end].strip()
            
            # Only add chunk if it contains content
            if chunk_text:
                # Create chunk entry with appropriate field names for the graph builder
                
                
                # Generate a document_id for the first chunk, then reuse for all chunks
                if chunk_id == 0:
                    document_uuid = self.datautility.generate_uuid()
                
                chunk = {
                    'source': markdown_file,
                    'document_id': document_uuid,
                    'chunk_id': self.datautility.generate_uuid(),
                    'document_name': filename,
                    'reference': f"{filename} Para {chunk_id + 1}.",
                    'hierarchy': filename,  # Just use filename, no heading hierarchy
                    'corpus': chunk_text,
                    'embedding_model': None
                }
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next chunk
            start = end - overlap
            
        # Convert to DataFrame
        df = pd.DataFrame(chunks)
        logger.info(f"Created {len(chunks)} chunks from {markdown_file}")
        return df

    def hierarchy_based_chunking(self, markdown_file: str, df_headings: pd.DataFrame) -> pd.DataFrame:
        """Extract hierarchical content chunks from a markdown file based on headings.
        
        Args:
            markdown_file: Path to the markdown file to process
            df_headings: DataFrame containing heading metadata with columns:
                - Level: Heading hierarchy level (e.g. 1, 2, 3)
                - Heading: Heading text
                - Page: Page number
                - File: File identifier (e.g. APS113)
                - Index: Index number
        
        Returns:
            DataFrame containing:
                - source: Source of the document
                - document_id: Unique identifier for source document
                - chunk_id: Unique identifier for text chunk
                - document_name: Name of the source document
                - hierarchy: Full heading path (e.g. "APS 113 > Main Body > Application")
                - heading: Current heading text
                - reference: Document reference
                - corpus: Text content under the heading
                - content_type: Type of content ('paragraph', 'table', etc.)
                - embedding_model: Model used to generate embeddings (None at this stage)
        """
        try:
            # Extract filename and read content
            filename = Path(markdown_file).stem.upper()
            logger.debug(f"Processing {filename}")
            
            # Clean heading metadata
            df_clean = df_headings.copy()
            df_clean['Level'] = pd.to_numeric(df_clean['Level'], errors='coerce', downcast='integer')
            df_clean['Heading'] = df_clean['Heading'].str.strip()
            df_clean = df_clean.dropna(subset=['Level', 'Heading'])
            
            # Filter headings for current document
            doc_headings = df_clean[df_clean['File'].str.contains(filename, case=False, na=False)]
            doc_headings = doc_headings.sort_values(by='Index')
            if doc_headings.empty:
                logger.error(f"No matching headings found for {filename}")
                raise

            # Load markdown file
            with open(markdown_file, 'r', encoding='utf-8') as f:
                md_content = f.read().strip()
            
            # Extract content after the first heading
            first_heading = "## " + doc_headings.iloc[0]['Heading']
            first_heading_index = md_content.find(first_heading)
            if first_heading_index != -1:
                # Extract date from content before first heading if exists
                # header_content = md_content[:first_heading_index]
                md_content = md_content[first_heading_index:]
                logger.debug(f"First heading {first_heading} found at index {first_heading_index}")
            else:
                md_content = ""
                logger.error(f"First heading {first_heading} not found in {filename}")
            
            # Split content into lines
            lines = md_content.split('\n')

            # Initialize variables
            extracted_sections = []
            current_heading = None
            current_content = []
            current_level = 0
            hierarchy = [] 
            
            # Process content line by line
            for i, line in enumerate(lines):                
                # Check if line is a heading
                if line.startswith('## '):
                    if current_heading:
                        self._append_section(filename = filename,
                                        hierarchy = hierarchy,
                                        current_heading = current_heading,
                                        current_content = current_content,
                                        extracted_sections = extracted_sections)
                    
                    current_heading = re.sub(r'\$.*\$', '', line[3:]).strip()
                    current_level = doc_headings[doc_headings['Heading'] == current_heading]['Level'].iloc[0]
                    doc_headings = doc_headings.drop(doc_headings.index[0])
                    
                    if current_level == 1:
                        if "Attachment" not in current_heading:
                            current_heading = filename + " - " + current_heading
                        hierarchy = [current_heading]
                    elif current_level >= 1:
                        if len(hierarchy) >= current_level:
                            hierarchy = hierarchy[:current_level-1]
                            hierarchy.append(current_heading)
                        else:
                            hierarchy.append(current_heading)  
                    current_content = []
                else:
                    if line.strip():
                        current_content.append(line.strip())
            
            # Append last section
            if current_heading:
                self._append_section(filename = filename,
                                hierarchy = hierarchy,
                                current_heading = current_heading,
                                current_content = current_content,
                                extracted_sections = extracted_sections)
            
            # Convert to DataFrame with proper column names
            if extracted_sections:                
                # Create base DataFrame
                df_sections = pd.DataFrame(extracted_sections, columns=['hierarchy', 'reference', 'heading', 'corpus'])
                
                # Add content_type column (default to 'paragraph')
                df_sections['content_type'] = 'paragraph'
                
                # Generate a single document_id for all chunks from this file
                document_uuid = self.datautility.generate_uuid()
                
                # Add required fields from schema
                df_sections['source'] = markdown_file
                df_sections['document_id'] = document_uuid
                df_sections['chunk_id'] = df_sections.apply(lambda _: self.datautility.generate_uuid(), axis=1)
                df_sections['document_name'] = filename
                df_sections['embedding_model'] = None
                
                logger.info(f"Extracted {len(df_sections)} sections from {markdown_file}")
                return df_sections
            else:
                # Return empty DataFrame with required columns
                logger.warning(f"No sections extracted from {markdown_file}")
                return pd.DataFrame(columns=['hierarchy', 'heading', 'reference', 'corpus', 'content_type', 
                                           'source', 'document_id', 'chunk_id', 'document_name', 'embedding_model'])
            
        except FileNotFoundError:
            logger.error(f"Markdown file not found: {markdown_file}")
            raise
        except Exception as e:
            logger.error(f"Error processing markdown file {markdown_file}: {str(e)}")
            raise

    def _append_section(
        self, 
        filename: str, 
        hierarchy: List[str], 
        current_heading: str, 
        current_content: List[str],
        extracted_sections: List[Dict[str, str]]):
        """ Append a section to the extracted sections.
        
        Args:
            filename: Document identifier
            hierarchy: Current heading hierarchy
            current_heading: Heading level of the current section
            current_content: Regular paragraph content
            extracted_sections: List to append the new section to
            aiutility: Utility object for AI-based content extraction
        """
        if len(hierarchy) == 1:
            hierarchy_heading = hierarchy[0]
        else:
            hierarchy_heading = ' > '.join(hierarchy)
        
        if len(current_content) == 1:
            text_contents = current_content[0]
        else:
            text_contents = '\n'.join(current_content)
        
        attachment_prefix = re.search(r'Attachment ([A-Z])', hierarchy_heading) or \
                            re.search(r'Chapter \d+', hierarchy_heading) or \
                            re.search(r'CHAPTER \d+', hierarchy_heading)
        prefix_para = ", ".join([filename, attachment_prefix.group(0)]) if attachment_prefix else filename

        # Specific to Basel Framework
        if "CRE" in filename and text_contents is not None:
            # Split by numbering system N.N
            paragraphs = re.split(r'\n(?=\d+\.\d+)', text_contents)
            for paragraph in paragraphs:
                match = re.search(r'(\d+\.\d+)', paragraph)
                if match:
                    para_num = match.group(2)
                    ref_num = ", CRE" + match.group(0)
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para + " Para " + para_num + ref_num, 
                                                current_heading,
                                                paragraph.strip()))
                else:
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para+ " Orphan", 
                                                current_heading,
                                                paragraph.strip()))
                logger.debug(f"Paragraph found: {filename} - {current_heading.strip()}")

        # Specific to APS, APG and Risk Opinions
        else:
            # APS/APG - Split by numbering system N. or Table M
            if re.search(r'^(\d+\.)|(Table \d+)', text_contents):
                paragraphs = re.split(r'\n(?=\d+\.|Table \d+)', text_contents)
            # Split for long paragraphs (based on character count instead of token count)
            elif len(text_contents) > 8000:
                paragraphs = re.split(r'\n', text_contents)
            else:
                paragraphs = [text_contents]

            for paragraph in paragraphs:
                match1 = re.match(r'^(\d+\.)', paragraph)
                match2 = re.match(r'^Table (\d+)', paragraph)
                if match1:
                    logger.debug(f"Paragraph found: {filename} - {current_heading.strip()}")
                    para_num = match1.group(1)
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para + " Para " + para_num, 
                                                current_heading,
                                                paragraph.strip()))
                elif match2:
                    logger.debug(f"Table found: {filename} - {current_heading.strip()}")
                    table_num = match2.group(1)
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para + " Table " + table_num, 
                                                current_heading,
                                                paragraph.strip()))
                else:
                    logger.debug(f"Orphan paragraph: {filename} - {current_heading.strip()}")
                    extracted_sections.append((filename + " > " + hierarchy_heading,
                                                prefix_para+ " Orphan", 
                                                current_heading,
                                                paragraph.strip()))


class VectorBuilder:
    """
    Creates and manages a vector database for knowledge representation.
    Depends on TextParser and TextChunker for text parsing and chunking.
    
    Methods:
        create_vectordb: Creates a vector database from markdown files
        load_vectordb: Loads a vector database from a file
        merge_vectordbs: Merges multiple vector databases into one
    """
    def __init__(self, parser: TextParser, chunker: TextChunker, generator=None, config_file_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize VectorBuilder with a DataCleanser instance.
        
        Parameters:
            parser (TextParser): Instance of TextParser.
            chunker (TextChunker): Instance of TextChunker.
            generator (Generator, optional): Instance of Generator. If None, a new one will be created.
            config_file_path (Optional[Union[str, Path]]): Path to the configuration file.
                If None, defaults to Path.cwd() / "config" / "main_config.json".
        
        Returns:
            None
        """
        # Store parser and chunker instances
        self.parser = parser
        self.chunker = chunker
        self.default_buffer_ratio = 0.9
        self.db_dir = Path.cwd() / "db"
        self.vector_dir = self.db_dir / "vector"
        self.vector_dir.mkdir(exist_ok=True, parents=True)
        
    def process_input_file(self, file_path: str, output_md_path: Optional[str] = None, conversion_method: str = 'pymupdf') -> str:
        """Process input file (PDF or Markdown) and return path to markdown file.
        
        Args:
            file_path: Path to input file (PDF or Markdown)
            output_md_path: Optional path for output markdown file (for PDF conversion)
            conversion_method: Method to use for PDF conversion ('pymupdf', 'openleaf', 'markitdown', 'ocr', 'llamaindex')
            
        Returns:
            str: Path to markdown file for further processing
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # If file is already markdown, return its path
        if file_ext == '.md':
            return file_path
        
        # If file is PDF, convert to markdown
        elif file_ext == '.pdf':
            if output_md_path is None:
                output_md_path = os.path.splitext(file_path)[0] + '.md'
                
            # Select conversion method
            if conversion_method == 'pymupdf':
                self.parser.pdf2md_pymupdf(file_path)
            elif conversion_method == 'openleaf':
                self.parser.pdf2md_openleaf(file_path)
            elif conversion_method == 'markitdown':
                self.parser.pdf2md_markitdown(file_path)
            elif conversion_method == 'ocr':
                self.parser.pdf2md_ocr(file_path, output_md_path)
            elif conversion_method == 'llamaindex':
                self.parser.pdf2md_llamaindex(file_path)
            else:
                raise ValueError(f"Unsupported conversion method: {conversion_method}")
                
            return output_md_path
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Only .md and .pdf are supported.")
        

    def apply_chunking(self, 
                      input_file: str,
                      df_headings: Optional[pd.DataFrame] = None,
                      chunking_method: str = 'hierarchy',
                      **kwargs) -> pd.DataFrame:
        """Apply text chunking to input file.
        
        Args:
            input_file: Path to the input file (PDF or markdown)
            df_headings: DataFrame containing heading metadata
            chunking_method: Method to use for chunking ('hierarchy' or 'length')
            **kwargs: Additional arguments for chunking method
            
        Returns:
            pd.DataFrame: DataFrame with chunked text
        """
        # Process input file (convert PDF to markdown if needed)
        markdown_file = self.process_input_file(input_file, **kwargs)
        
        # Validate chunking method
        if chunking_method not in ['hierarchy', 'length']:
            raise ValueError(f"Invalid chunking method: {chunking_method}")
        
        # Set defaults and validate parameters based on chunking method
        if chunking_method == 'length':
            chunk_size = kwargs.get('chunk_size', self.default_chunk_size)
            chunk_overlap = kwargs.get('chunk_overlap', self.default_chunk_overlap)
        else:  # hierarchy based chunking
            if any(k in kwargs for k in ['chunk_size', 'chunk_overlap']):
                logger.warning("chunk_size and chunk_overlap are ignored for hierarchy-based chunking")
        
        # Get list of markdown files and hierarchy data
        if not markdown_file:
            raise ValueError(f"Markdown file not provided for vector database building")
        if df_headings is not None and not df_headings.empty:
            # Choose chunking method
            if chunking_method == 'hierarchy':
                chunks_df = self.chunker.hierarchy_based_chunking(
                    markdown_file,
                    df_headings)
            else:
                chunks_df = self.chunker.length_based_chunking(
                    markdown_file,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )      
            return chunks_df
        else:
            raise ValueError(f"Hierarchy data not provided for vector database building")

    def create_embeddings(self, 
                          chunks_df: pd.DataFrame,
                          model: Optional[str] = None,
                          **kwargs) -> pd.DataFrame:
        """Generate embeddings for text chunks.
        
        Args:
            chunks_df: DataFrame with chunked text
            model: Model to use for embedding generation
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            pd.DataFrame: DataFrame with embeddings added
        """
        # Generate embeddings for corpus and hierarchy
        logger.info("Generating embeddings for corpus texts")
        buffer_ratio = kwargs.get('buffer_ratio', self.default_buffer_ratio)
        
        corpus_vectors = self.generator.get_embeddings(
            text=chunks_df['corpus'].tolist(),
            model=model,
            buffer_ratio=buffer_ratio
        )
        
        logger.info("Generating embeddings for hierarchy texts")
        hierarchy_vectors = self.generator.get_embeddings(
            text=chunks_df['hierarchy'].tolist(),
            model=model,
            buffer_ratio=buffer_ratio
        )
        
        # Add vector columns
        chunks_df['corpus_vector'] = corpus_vectors
        chunks_df['hierarchy_vector'] = hierarchy_vectors
        
        # Update embedding_model field if not already set
        if 'embedding_model' in chunks_df.columns and chunks_df['embedding_model'].isnull().all():
            chunks_df['embedding_model'] = model
        
        return chunks_df


    def save_db(self, 
                chunks_df: pd.DataFrame,
                input_file: str,
                **kwargs) -> str:
        """Save DataFrame to parquet file.
        
        Args:
            chunks_df: DataFrame to save
            input_file: Original input file path (used for naming)
            **kwargs: Additional arguments for parquet saving
            
        Returns:
            str: Path to saved parquet file
        """
        # Save to parquet file
        input_name_no_ext = os.path.splitext(os.path.basename(input_file))[0].lower()
        
        # Ensure using proper db directory path
        output_file = self.vector_dir / f'v_{input_name_no_ext}.parquet'
        chunks_df.to_parquet(output_file)
        
        logger.info(f"Vector database created with {len(chunks_df)} chunks and saved to {output_file}")
        return str(output_file)


    def create_db(self, 
                  input_file: str,
                  df_headings: pd.DataFrame,
                  chunking_method: str = 'hierarchy',
                  model: Optional[str] = None,
                  **kwargs) -> str:
        """Create a vector database from input files.
        
        Args:
            input_file: Path to the input file (PDF or markdown)
            df_headings: DataFrame containing heading metadata
            chunking_method: Method to use for chunking ('hierarchy' or 'length')
            model: Model to use for embedding generation
            **kwargs: Additional arguments based on chunking method
            
        Returns:
            str: Path to the saved vector database file
        """
        try:
            # Step 1: Apply chunking
            chunks_df = self.apply_chunking(
                input_file=input_file,
                df_headings=df_headings,
                chunking_method=chunking_method,
                **kwargs  # e.g. conversion_method
            )

            # Step 2: Create embeddings
            chunks_df = self.create_embeddings(
                chunks_df=chunks_df,
                model=model,
                **kwargs
            )
            
            # Step 3: Save to parquet
            output_file = self.save_db(
                chunks_df=chunks_df,
                input_file=input_file,
                **kwargs
            )
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error creating vector database: {str(e)}")
            logger.debug(f"Vector database creation error details: {traceback.format_exc()}")
            raise

    def load_db(self, parquet_file: str = None) -> pd.DataFrame:
        """Load the vector database from disk.
        
        Args:
            parquet_file: Path to the parquet file containing the vector database.
                          If None, will look for a file in the db directory.
            
        Returns:
            DataFrame containing the knowledge base with vector columns
            
        Raises:
            FileNotFoundError: If the vector database file doesn't exist
            ValueError: If the file format is not supported or the file is corrupted
        """
        logger.debug(f"Starting to load vector database from {parquet_file}")
        start_time = time.time()
        
        try:
            # If no filepath is provided, try to find a vector database in the db/vector directory
            if parquet_file is None:
                vector_files = list(self.vector_dir.glob('v_*.parquet'))
                if not vector_files:
                    raise FileNotFoundError("No vector database files found in the db directory")
                # Use the most recently modified file
                parquet_file = str(sorted(vector_files, key=lambda f: f.stat().st_mtime, reverse=True)[0])
                logger.debug(f"Using most recent vector database file: {parquet_file}")
            
            # Check if file exists
            if not os.path.exists(parquet_file):
                raise FileNotFoundError(f"Vector database not found: {parquet_file}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_file)
            
            # Validate required columns
            required_columns = ['reference', 'hierarchy', 'corpus']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in vector database")
            
            # Check for vector columns
            vector_columns = [col for col in df.columns if 'vector' in col or 'embedding' in col]
            if not vector_columns:
                logger.warning(f"No vector columns found in {parquet_file}")
            
            # Convert vector columns from string to numpy arrays if needed
            for col in vector_columns:
                if df[col].dtype == object:
                    try:
                        df[col] = df[col].apply(np.array)
                    except Exception as e:
                        logger.warning(f"Could not convert column {col} to numpy arrays: {e}")
            
            logger.info(f"Loaded vector database from {parquet_file} with {len(df)} chunks in {time.time() - start_time:.2f} seconds")
            return df
            
        except Exception as e:
            logger.error(f"Error loading vector database: {str(e)}")
            logger.debug(f"Load error details: {traceback.format_exc()}")
            raise
    
    def merge_db(self, parquet_files: List[str], output_name: str = None) -> pd.DataFrame:
        """Merge multiple vector databases into one.
        
        Args:
            parquet_files: List of paths to vector database files to merge
            output_name: Name for the merged vector database file (without extension)
            
        Returns:
            pd.DataFrame: The merged vector database
            
        Raises:
            ValueError: If no parquet_files are provided or they are incompatible
            FileNotFoundError: If any of the specified files don't exist
        """
        logger.debug(f"Starting to merge {len(parquet_files)} vector databases")
        start_time = time.time()
        
        if not parquet_files:
            raise ValueError("No parquet files provided for merging")
            
        try:
            # Load and validate each dataframe
            dataframes = []
            for i, filepath in enumerate(parquet_files):
                logger.debug(f"Loading vector database {i+1}/{len(parquet_files)}: {filepath}")
                
                # Check if file exists
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Vector database file not found: {filepath}")
                
                # Load dataframe
                df = pd.read_parquet(filepath)
                
                # Validate required columns
                required_columns = ['reference', 'hierarchy', 'corpus']
                for col in required_columns:
                    if col not in df.columns:
                        logger.warning(f"Required column '{col}' not found in {filepath}, skipping")
                        continue
                
                # Check for vector columns
                vector_columns = [col for col in df.columns if 'vector' in col or 'embedding' in col]
                if not vector_columns:
                    logger.warning(f"No vector columns found in {filepath}, skipping")
                    continue
                
                # Add prefix to reference IDs to avoid conflicts
                prefix = f"db{i}_"
                df['reference'] = df['reference'].apply(lambda x: f"{prefix}{x}")
                
                # Add source file information
                df['source_file'] = os.path.basename(filepath)
                
                dataframes.append(df)
                logger.debug(f"Added {len(df)} chunks from {filepath}")
            
            if not dataframes:
                raise ValueError("No valid vector databases found to merge")
                
            # Merge dataframes
            merged_df = pd.concat(dataframes, ignore_index=True)
            
            # Generate output name if not provided
            if not output_name:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"merged_vectordb_{timestamp}"
                
            # Remove .parquet extension if included
            if output_name.endswith('.parquet'):
                output_name = output_name[:-8]
                
            # Save the merged vector database
            output_path = self.vector_dir / f"v_{output_name}.parquet"
            merged_df.to_parquet(output_path)
                
            logger.info(f"Merged {len(dataframes)} vector databases into {output_path} with {len(merged_df)} total chunks in {time.time() - start_time:.2f} seconds")
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to merge vector databases: {str(e)}")
            logger.debug(f"Merge error details: {traceback.format_exc()}")
            raise
    

class MemoryBuilder:
    """
    Builds the memory database by extracting entries from the vector database.
    Initializes dummy entries from the vector database for episodic memory.
    
    Methods:
        create_db: Creates a memory database from vector database entries
        load_db: Loads a memory database from a file
    """
    
    def __init__(self) -> None:
        """
        Initialize a MemoryBuilder instance.
        """
        try:
            self.datautility = DataUtility()
            # Set up database directory paths
            self.db_dir = Path.cwd() / "db"
            self.memory_dir = self.db_dir / "memory"
            self.vector_dir = self.db_dir / "vector"
            
            # Create directories if they don't exist
            self.memory_dir.mkdir(exist_ok=True, parents=True)
            logger.debug(f"MemoryBuilder initialized with memory_dir: {self.memory_dir}")
        except Exception as e:
            logger.error(f"MemoryBuilder initialization failed: {str(e)}")
            logger.debug(f"Initialization error details: {traceback.format_exc()}")
            raise
    
    def create_db(self, vector_db_file: Optional[str] = None, num_entries: int = 5) -> str:
        """
        Create a memory database from vector database entries.
        Extracts the first N entries from a vector database and stores them as
        episodic memory entries in the memory database.
        
        Args:
            vector_db_file: Path to the vector database parquet file.
                           If None, will look for a file in the db/vector directory.
            num_entries: Number of entries to extract from the vector database (default: 5)
            
        Returns:
            str: Path to the saved memory database file
            
        Raises:
            FileNotFoundError: If the vector database file doesn't exist
            ValueError: If the vector database format is not supported or the file is corrupted
        """
        logger.debug(f"Starting to create memory database from {vector_db_file}")
        start_time = time.time()
        
        try:
            # If no filepath is provided, try to find a vector database in the db/vector directory
            if vector_db_file is None:
                vector_files = list(self.vector_dir.glob('v_*.parquet'))
                if not vector_files:
                    raise FileNotFoundError("No vector database files found in the db/vector directory")
                # Use the most recently modified file
                vector_db_file = str(sorted(vector_files, key=lambda f: f.stat().st_mtime, reverse=True)[0])
                logger.debug(f"Using most recent vector database file: {vector_db_file}")
            
            # Check if file exists
            if not os.path.exists(vector_db_file):
                raise FileNotFoundError(f"Vector database not found: {vector_db_file}")
            
            # Load vector database
            df_vector = pd.read_parquet(vector_db_file)
            
            # Validate required columns
            required_columns = ['corpus', 'document_name', 'hierarchy']
            for col in required_columns:
                if col not in df_vector.columns:
                    raise ValueError(f"Required column '{col}' not found in vector database")
            
            # Extract the first N entries
            sample_df = df_vector.head(num_entries).copy()
            
            # Create memory entries conforming to the memory_db schema
            memory_entries = []
            for i, (_, row) in enumerate(sample_df.iterrows(), 1):
                memory_entry = {
                    'memory_id': self.datautility.generate_uuid(),
                    'query': f"Tell me about {row['document_name']}",
                    'entity': f"Dummy Entity {i}",
                    'context': {
                        'document_name': row['document_name'],
                        'hierarchy': row['hierarchy'],
                        'content': row['corpus'],
                        'source': "Dummy Source",
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                }
                memory_entries.append(memory_entry)
            
            # Convert to DataFrame
            memory_df = pd.DataFrame(memory_entries)
            
            # Save to parquet file in the memory directory            
            output_file = self.memory_dir / f'memory.parquet'
            memory_df.to_parquet(output_file)
            
            logger.info(f"Memory database created with {len(memory_df)} entries and saved to {output_file}")
            logger.debug(f"Memory database creation completed in {time.time() - start_time:.2f} seconds")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error creating memory database: {str(e)}")
            logger.debug(f"Memory database creation error details: {traceback.format_exc()}")
            raise
    
    def load_db(self, memory_db_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load the memory database from disk.
        
        Args:
            memory_db_file: Path to the parquet file containing the memory database.
                           If None, will look for a file in the db/memory directory.
            
        Returns:
            DataFrame containing the memory database
            
        Raises:
            FileNotFoundError: If the memory database file doesn't exist
            ValueError: If the file format is not supported or the file is corrupted
        """
        logger.debug(f"Starting to load memory database from {memory_db_file}")
        start_time = time.time()
        
        try:
            # If no filepath is provided, try to find a memory database in the db/memory directory
            if memory_db_file is None:
                memory_files = list(self.memory_dir.glob('memory.parquet'))
                if not memory_files:
                    raise FileNotFoundError("No memory database files found in the db/memory directory")
                # Use the most recently modified file
                memory_db_file = str(sorted(memory_files, key=lambda f: f.stat().st_mtime, reverse=True)[0])
                logger.debug(f"Using most recent memory database file: {memory_db_file}")
            
            # Check if file exists
            if not os.path.exists(memory_db_file):
                raise FileNotFoundError(f"Memory database not found: {memory_db_file}")
            
            # Load parquet file
            df = pd.read_parquet(memory_db_file)
            
            # Validate required columns
            required_columns = ['memory_id', 'query', 'entity', 'context']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in memory database")
            
            logger.info(f"Loaded memory database from {memory_db_file} with {len(df)} entries in {time.time() - start_time:.2f} seconds")
            return df
            
        except Exception as e:
            logger.error(f"Error loading memory database: {str(e)}")
            logger.debug(f"Load error details: {traceback.format_exc()}")
            raise
    

class GraphBuilder:
    """
    Builds graph representations of the knowledge base using NetworkX.
    Creates graph databases that conform to the schema defined in schema_graph_db.json.
    
    The graph database stores structural relationships between text chunks, documents,
    and sections, with references to content stored in the vector database.
    
    Methods:
        create_db: Creates a graph database from a vector database
        save_db: Saves the graph to a file in the db/graph directory
        load_db: Loads a graph from a file in the db/graph directory
        merge_dbs: Merges multiple graph databases into one
        view_db: Generates visual representations of the graphs
    """

    def __init__(self, vectordb_file: Optional[str] = None) -> None:
        """Initialize a GraphBuilder instance.
        
        Args:
            vectordb_file (Optional[str]): Path to the vector database file to use.
                If provided, relative paths are resolved relative to the db/vector directory.
                If None, no database is loaded initially.
        """
        try:
            # Initialize NetworkX for graph operations
            self.nx = nx
            self.datautility = DataUtility()
            
            # Initialize generator and meta-generator for LLM operations
            self.generator = Generator()
            self.metagenerator = MetaGenerator(generator=self.generator)
            self.aiutility = AIUtility()
            
            # Set up directory structure
            self.db_dir = Path.cwd() / "db"
            self.graph_dir = self.db_dir / "graph"
            self.vector_dir = self.db_dir / "vector"
            
            # Create directories if they don't exist
            self.graph_dir.mkdir(exist_ok=True, parents=True)
            
            # Store the base vectordb file path
            if vectordb_file:
                # Handle both relative and absolute paths
                if os.path.isabs(vectordb_file):
                    self.vectordb_path = Path(vectordb_file)
                else:
                    self.vectordb_path = self.vector_dir / vectordb_file
                
                self.db_name = Path(vectordb_file).stem
                if self.db_name.startswith('v_'):
                    self.db_name = self.db_name[2:]  # Remove 'v_' prefix
                logger.debug(f"GraphBuilder using vector database file: {self.vectordb_path}")
            else:
                self.vectordb_path = None
                self.db_name = None
                logger.debug("GraphBuilder initialized without a base vector database file")
            
            # Initialize graph containers
            self.graph = None  # Standard graph conforming to schema
            self.hypergraph = None  # Hypergraph for advanced queries
            
            # Track creation timestamp for metadata
            self.created_at = datetime.datetime.now().isoformat()
            
            logger.debug("GraphBuilder initialized successfully")
        except ImportError:
            logger.error("NetworkX not found. Please install with: pip install networkx")
            raise

    def create_db(self, vector_db_file: Optional[str] = None, graph_type: str = 'standard') -> nx.Graph:
        """
        Create a graph database from a vector database file.
        
        This method builds a graph representation of the knowledge base that conforms to
        the schema defined in schema_graph_db.json. The process involves the following steps:
        
        1. Determine the source vector database file to use, with fallback mechanisms
        2. Load and validate the vector database content
        3. Create the appropriate graph structure based on the specified type:
           - 'standard': Creates a directed graph with hierarchical relationships
           - 'hypergraph': Creates a hypergraph with additional semantic relationships
        4. Store the graph in the instance and return it
        
        The resulting graph will contain the following node types:
        - Document: Root nodes representing source documents
        - Section: Intermediate nodes representing document sections
        - Chunk: Leaf nodes representing text chunks with embedded content
        
        Relationships between nodes include:
        - CONTAINS: Document → Section → Chunk hierarchy
        - REFERENCES: Cross-references between related nodes
        - SIMILAR: Similarity relationships between chunks
        
        Args:
            vector_db_file: Path to the vector database parquet file. If None, uses the file
                           specified during initialization or finds the most recent one.
            graph_type: Type of graph to create ('standard' or 'hypergraph')
            
        Returns:
            nx.Graph: The created graph database with nodes and relationships
            
        Raises:
            FileNotFoundError: If the vector database file doesn't exist
            ValueError: If the vector database format is not supported or the file is corrupted
            
        Example:
            >>> builder = GraphBuilder()
            >>> graph = builder.create_db('my_vectors.parquet', graph_type='standard')
            >>> print(f"Created graph with {graph.number_of_nodes()} nodes")
        """
        logger.debug(f"Starting to create {graph_type} graph database")
        start_time = time.time()
        
        try:
            # Step 1: Determine the vector database file to use
            # Priority: 1. Explicitly provided file 2. Instance's vectordb_path 3. Most recent file in vector_dir
            if vector_db_file is not None:
                # Handle both relative and absolute paths
                if os.path.isabs(vector_db_file):
                    vector_path = Path(vector_db_file)
                else:
                    vector_path = self.vector_dir / vector_db_file
            elif self.vectordb_path is not None:
                vector_path = self.vectordb_path
            else:
                # Try to find a vector database file
                vector_files = list(self.vector_dir.glob('v_*.parquet'))
                if not vector_files:
                    raise FileNotFoundError("No vector database files found in the db/vector directory")
                # Use the most recently modified file
                vector_path = sorted(vector_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
                logger.debug(f"Using most recent vector database file: {vector_path}")
            
            # Check if file exists
            if not os.path.exists(vector_path):
                raise FileNotFoundError(f"Vector database not found: {vector_path}")
            
            # Step 2: Load and validate the vector database
            logger.debug(f"Loading vector database from {vector_path}")
            df_vector = pd.read_parquet(vector_path)
            
            # Step 3: Validate the vector database structure
            # Ensure all required columns are present for graph construction
            required_columns = ['document_id', 'chunk_id', 'document_name', 'hierarchy', 'corpus']
            for col in required_columns:
                if col not in df_vector.columns:
                    raise ValueError(f"Required column '{col}' not found in vector database")
            
            # Step 4: Create the appropriate graph based on type
            logger.debug(f"Creating {graph_type} graph structure")
            if graph_type == 'standard':
                # Standard graph with hierarchical relationships
                graph = self._create_standard_graph(df_vector)
                self.graph = graph  # Store reference to the standard graph
            elif graph_type == 'hypergraph':
                # Hypergraph with additional semantic relationships
                graph = self._create_hypergraph(df_vector)
                self.hypergraph = graph  # Store reference to the hypergraph
            else:
                raise ValueError(f"Unknown graph type: {graph_type}")
                
            # Step 5: Store metadata about the graph
            self._store_graph_metadata(graph, graph_type)
            
            # Save the database name for future reference
            self.db_name = Path(vector_path).stem
            if self.db_name.startswith('v_'):
                self.db_name = self.db_name[2:]  # Remove 'v_' prefix
            
            logger.info(f"Created {graph_type} graph database with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges in {time.time() - start_time:.2f} seconds")
            return graph
            
        except Exception as e:
            logger.error(f"Failed to create graph database: {str(e)}")
            logger.debug(f"Graph creation error details: {traceback.format_exc()}")
            raise
    
    def _find_cross_references(self, chunk_content: str, all_chunks: pd.DataFrame) -> list:
        """
        Extract cross-references from chunk content using an open-source language model.
        
        This method analyzes the content of a text chunk to identify references to other chunks
        in the corpus. It uses a language model with a specialized meta prompt to extract
        these references.
        
        Args:
            chunk_content: The text content of the chunk to analyze
            all_chunks: DataFrame containing all chunks in the corpus for reference matching
            
        Returns:
            list: List of chunk_ids that are referenced by this chunk
        """
        try:
            logger.debug("Extracting cross-references using MetaGenerator")
            # Use the metagenerator to apply the meta prompt and call the language model
            try:                
                # Call metagenerator with the custom meta prompt
                # This follows the pattern seen in topologist.py examples
                response = self.metagenerator.get_meta_generation(
                    application="metaworkflow",  # Custom application since we're not using a library prompt
                    category="dbbuilder",  # Category for extraction tasks
                    action="crossreference",  # Specific action
                    prompt_id=100,  # Default prompt ID
                    model="Qwen2.5-1.5B",  # Use a smaller, efficient model
                    temperature=0.7,  # Balanced temperature for creative yet focused extraction
                    max_tokens=500,  # Reasonable length for responses
                    content=chunk_content,  # Pass the chunk content as a parameter
                    return_full_response=False  # We just want the text output
                )
                
                # Parse the response to extract references
                try:
                    if response:
                        references_data = self.aiutility.format_json_response(response)
                        references = references_data.get("references", [])
                    else:
                        references = []
                except (ValueError, TypeError) as je:
                    logger.warning(f"Could not parse JSON from LLM response: {str(je)}")
                    references = []
            except Exception as e:
                logger.warning(f"Error using MetaGenerator: {str(e)}. Using fallback method.")
                references = []
            
            # Match references to actual chunks in the corpus - this part remains mostly unchanged
            referenced_chunk_ids = []
            if references and len(references) > 0:
                # Create a function to score how well a reference matches a chunk
                def match_score(reference, row):
                    # Check various fields for matches
                    score = 0
                    reference = reference.lower()
                    
                    # Check content (most important)
                    if 'content' in row and isinstance(row['content'], str):
                        content = row['content'].lower()
                        if reference in content:
                            score += 10  # Strong match in content
                        
                    # Check document name and section headers
                    if 'document_name' in row and isinstance(row['document_name'], str):
                        doc_name = row['document_name'].lower()
                        if reference in doc_name:
                            score += 5  # Good match in document name
                    
                    # Check section hierarchy
                    if 'hierarchy' in row and isinstance(row['hierarchy'], str):
                        hierarchy = row['hierarchy'].lower()
                        if reference in hierarchy:
                            score += 3  # Match in section hierarchy
                    
                    return score
                
                # For each reference, find the best matching chunks
                for ref in references:
                    if not ref or len(ref.strip()) < 3:  # Skip very short references
                        continue
                        
                    # Calculate match scores for all chunks
                    all_chunks['match_score'] = all_chunks.apply(lambda row: match_score(ref, row), axis=1)
                    
                    # Get the top matches (score > 0 and max 3 per reference)
                    matches = all_chunks[all_chunks['match_score'] > 0].nlargest(3, 'match_score')
                    
                    # Add the chunk IDs to our result list
                    for _, match in matches.iterrows():
                        if match['chunk_id'] not in referenced_chunk_ids:  # Avoid duplicates
                            referenced_chunk_ids.append(match['chunk_id'])
                
                # Remove the temporary column we added
                if 'match_score' in all_chunks.columns:
                    all_chunks.drop('match_score', axis=1, inplace=True)
            
            logger.debug(f"Found {len(referenced_chunk_ids)} cross-references for chunk")
            return referenced_chunk_ids
            
        except Exception as e:
            logger.warning(f"Failed to extract cross-references: {str(e)}")
            logger.debug(f"Cross-reference extraction error details: {traceback.format_exc()}")
            return []
    
    # _find_similar_chunks is implemented above (consolidation of methods)

    def _create_standard_graph(self, df_vector: pd.DataFrame) -> nx.MultiDiGraph:
        """
        Create a standard graph database conforming to schema_graph_db.json.
        
        This internal method builds a directed multigraph with nodes for documents,
        sections, and text chunks, with relationships between them as defined in the schema.
        
        Args:
            df_vector: DataFrame containing vector database entries
            
        Returns:
            nx.MultiDiGraph: The created standard graph
        """
        logger.debug("Creating standard graph database")
        
        # Create a new directed multigraph
        G = self.nx.MultiDiGraph()
        
        # Track created nodes to avoid duplicates
        created_documents = set()
        created_sections = set()
        created_chunks = set()
        
        # Process each row in the DataFrame
        for _, row in df_vector.iterrows():
            # Create Chunk node (required by schema)
            chunk_id = row['chunk_id']
            if chunk_id not in created_chunks:
                G.add_node(chunk_id, 
                          node_type='Chunk',
                          chunk_id=chunk_id,
                          text=row['corpus'],  # Added chunk content
                          created_at=self.created_at)
                created_chunks.add(chunk_id)
            
            # Create Document node if it doesn't exist
            document_id = row['document_id']
            if document_id not in created_documents:
                G.add_node(document_id, 
                          node_type='Document',
                          document_id=document_id,
                          created_at=self.created_at)
                created_documents.add(document_id)
            
            # Process hierarchy to create Section nodes
            hierarchy_parts = row['hierarchy'].split(' > ')
            current_level = 1
            parent_section = None
            
            for i, part in enumerate(hierarchy_parts):
                # Generate a consistent section ID based on document and hierarchy path
                section_path = ' > '.join(hierarchy_parts[:i+1])
                section_id = f"section_{document_id}_{hash(section_path) % 10000000}"
                
                if section_id not in created_sections:
                    G.add_node(section_id,
                              node_type='Section',
                              section_id=section_id,
                              section_name=part,
                              level=current_level,
                              parent_document_id=document_id,
                              section_order=i,
                              created_at=self.created_at)
                    created_sections.add(section_id)
                    
                    # Connect to document
                    if i == 0:  # Top-level section
                        G.add_edge(document_id, section_id, 
                                  edge_type='CONTAINS', 
                                  order=i,
                                  created_at=self.created_at)
                    # Connect to parent section
                    elif parent_section:
                        G.add_edge(parent_section, section_id, 
                                  edge_type='CONTAINS', 
                                  order=i,
                                  created_at=self.created_at)
                
                parent_section = section_id
                current_level += 1
            
            # Connect the deepest section to the text chunk
            if parent_section:
                G.add_edge(parent_section, chunk_id, 
                          edge_type='CONTAINS', 
                          order=0,  # Only one chunk per section-chunk relationship
                          created_at=self.created_at)
            else:  # Fallback: connect document directly to chunk
                G.add_edge(document_id, chunk_id, 
                          edge_type='CONTAINS', 
                          order=0,
                          created_at=self.created_at)
            
            # Extract cross-references from chunk content using LLM
            if 'content' in row and row['content']:
                # Extract cross-references using LLM-based analysis
                referenced_chunks = self._find_cross_references(row['content'], df_vector)
                
                # Create REFERENCES edges for each extracted cross-reference
                for ref_chunk_id in referenced_chunks:
                    if ref_chunk_id in created_chunks and ref_chunk_id != chunk_id:
                        G.add_edge(chunk_id, ref_chunk_id, 
                                  edge_type='REFERENCES',
                                  reference_type='cross_reference',  # Using schema-defined reference type
                                  created_at=self.created_at,
                                  created_by='llm')
            
            # Add similarity relationships if corpus_vector is available (required by schema)
            # This enforces the schema requirement that corpus_vector must be present
            if 'corpus_vector' in row and isinstance(row['corpus_vector'], (list, np.ndarray)):
                similar_chunks = self._find_similar_chunks(row, df_vector)
                for similar_id, similarity in similar_chunks:
                    if similar_id in created_chunks and similar_id != chunk_id:
                        G.add_edge(chunk_id, similar_id, 
                                  edge_type='SIMILAR',
                                  similarity_score=float(similarity),
                                  similarity_type='semantic',
                                  embedding_model=row.get('embedding_model', 'unknown'),
                                  threshold=0.7,
                                  created_at=self.created_at)
        
        return G

    def _create_hypergraph(self, df_vector: pd.DataFrame) -> nx.Graph:
        """
        Create a hypergraph representation for advanced queries.
        
        This internal method builds a bipartite graph to represent hyperedges connecting
        multiple nodes based on document groups, topic clusters, and hierarchical levels.
        It also creates direct edges for cross-references and similarities between chunks.
        
        Args:
            df_vector: DataFrame containing vector database entries
            
        Returns:
            nx.Graph: Bipartite graph representing the hypergraph with additional direct relationships
        """
        logger.debug("Creating hypergraph database")
        
        # Create a bipartite graph to represent the hypergraph
        H = self.nx.Graph()
        
        # Track hyperedge IDs and created nodes
        next_edge_id = 0
        created_chunks = set()
        
        # First, create all chunk nodes
        for _, row in df_vector.iterrows():
            chunk_id = row['chunk_id']
            if chunk_id not in created_chunks:
                H.add_node(chunk_id, 
                           node_type='Chunk',
                           chunk_id=chunk_id,
                          document_id=row['document_id'],
                          document_name=row['document_name'],
                          created_at=self.created_at)
                created_chunks.add(chunk_id)
        
        # Create document group hyperedges
        doc_groups = df_vector.groupby('document_id')
        for doc_id, group in doc_groups:
            edge_id = f"he_doc_{next_edge_id}"
            next_edge_id += 1
            H.add_node(edge_id, type='hyperedge', edge_type='document_group')
            
            # Connect all chunks in this document to the hyperedge
            for _, row in group.iterrows():
                H.add_edge(edge_id, row['chunk_id'])
        
        # Create topic cluster groups using corpus_vector (implementing TOPIC_GROUP relationship)
        if 'corpus_vector' in df_vector.columns:
            import uuid
            # Generate clusters using the improved clustering method with corpus_vector
            clusters = self._cluster_by_embedding(df_vector, vector_field='corpus_vector')
            
            for cluster_idx, cluster_chunks in enumerate(clusters):
                if len(cluster_chunks) > 1:  # Only create groups for actual clusters
                    # Create a hyperedge node for the topic cluster
                    edge_id = f"he_topic_{next_edge_id}"
                    next_edge_id += 1
                    
                    # Generate required properties for TOPIC_GROUP
                    topic_id = str(uuid.uuid4())
                    topic_name = f"Topic Cluster {cluster_idx + 1}"
                    # Extract keywords from chunks in this cluster
                    topic_keywords = self._extract_topic_keywords(cluster_chunks, df_vector)
                    # Calculate coherence score for the cluster
                    coherence_score = self._calculate_cluster_coherence(cluster_chunks, df_vector)
                    
                    # Add hyperedge node with topic metadata
                    H.add_node(edge_id, 
                               type='hyperedge', 
                               edge_type='topic_group',
                               topic_id=topic_id,
                               topic_name=topic_name,
                               topic_keywords=topic_keywords,
                               coherence_score=coherence_score,
                               algorithm='hdbscan',  # Using HDBSCAN for clustering
                               created_at=self.created_at)
                    
                    # Connect chunks to the topic hyperedge with membership scores
                    for chunk_id in cluster_chunks:
                        if chunk_id in created_chunks:  # Ensure chunk exists
                            # Calculate membership score for this chunk in the cluster
                            membership_score = self._calculate_membership_score(chunk_id, cluster_idx, clusters, df_vector)
                            
                            # Add edge with required TOPIC_GROUP properties
                            H.add_edge(edge_id, chunk_id,
                                       type='TOPIC_GROUP',
                                       topic_id=topic_id,
                                       membership_score=membership_score)
        
        # Create hierarchy level hyperedges
        # Extract level from hierarchy depth
        df_vector['level'] = df_vector['hierarchy'].apply(lambda h: len(h.split(' > ')))
        level_groups = df_vector.groupby('level')
        
        for level, group in level_groups:
            edge_id = f"he_level_{next_edge_id}"
            next_edge_id += 1
            H.add_node(edge_id, type='hyperedge', edge_type='hierarchy_level', level=level)
            
            for _, row in group.iterrows():
                H.add_edge(edge_id, row['chunk_id'])
        
        # Add direct cross-reference relationships between chunks
        for _, row in df_vector.iterrows():
            chunk_id = row['chunk_id']
            
            # Extract cross-references from chunk content using LLM
            if 'content' in row and row['content']:
                # Extract cross-references using LLM-based analysis
                referenced_chunks = self._find_cross_references(row['content'], df_vector)
                
                # Create REFERENCES edges for each extracted cross-reference
                for ref_chunk_id in referenced_chunks:
                    if ref_chunk_id in created_chunks and ref_chunk_id != chunk_id:
                        # Add a direct edge with REFERENCES type
                        H.add_edge(chunk_id, ref_chunk_id, 
                                  type='REFERENCES',
                                  reference_type='derived',  # This was derived by LLM analysis
                                  created_at=self.created_at,
                                  created_by='llm')
        
        # Add similarity relationships based on corpus vectors
        for _, row in df_vector.iterrows():
            chunk_id = row['chunk_id']
            
            # Use corpus_vector for similarity if available
            if 'corpus_vector' in row and isinstance(row['corpus_vector'], (list, np.ndarray)):
                similar_chunks = self._find_similar_chunks(row, df_vector)
                for similar_id, similarity in similar_chunks:
                    if similar_id in created_chunks and similar_id != chunk_id:
                        # Add a direct edge with SIMILAR type
                        H.add_edge(chunk_id, similar_id, 
                                  edge_type='SIMILAR',
                                  similarity_score=float(similarity),
                                  similarity_type='corpus_semantic',
                                  embedding_model=row.get('embedding_model', 'unknown'),
                                  created_at=self.created_at)
        
        return H
    

    def _find_similar_chunks(self, row: pd.Series, df: pd.DataFrame, threshold: float = 0.7, top_k: int = 10) -> list:
        """
        Find chunks with similar embeddings, prioritizing corpus_vector field.
        
        This unified method handles both corpus_vector and embedding fields,
        with a preference for corpus_vector when available.
        
        Args:
            row: DataFrame row containing the source chunk and its vector
            df: DataFrame containing all chunks
            threshold: Minimum similarity score to consider (default: 0.7)
            top_k: Maximum number of similar chunks to return (default: 10)
            
        Returns:
            List of tuples containing (chunk_id, similarity_score)
        """
        try:
            # Check if corpus_vector is available, otherwise fall back to embedding
            vector_field = 'corpus_vector' if 'corpus_vector' in row and isinstance(row['corpus_vector'], (list, np.ndarray)) else 'embedding'
            
            if vector_field not in row or not isinstance(row[vector_field], (list, np.ndarray)):
                logger.warning(f"No valid {vector_field} found for similarity comparison")
                return []
            
            # Convert to numpy array and normalize
            query_vector = np.array(row[vector_field])
            query_vector = query_vector / np.linalg.norm(query_vector)
            
            # Calculate similarities for all chunks with appropriate vectors
            similarities = []
            for _, other_row in df.iterrows():
                if other_row['chunk_id'] == row['chunk_id']:
                    continue  # Skip self
                
                # Use the same vector field as the query
                if vector_field not in other_row or not isinstance(other_row[vector_field], (list, np.ndarray)):
                    continue  # Skip if no matching vector
                
                # Convert to numpy array and normalize
                other_vector = np.array(other_row[vector_field])
                other_vector = other_vector / np.linalg.norm(other_vector)
                
                # Calculate cosine similarity
                try:
                    similarity = np.dot(query_vector, other_vector)
                    if similarity >= threshold:
                        similarities.append((other_row['chunk_id'], similarity))
                except Exception as e:
                    logger.debug(f"Error calculating similarity: {str(e)}")
                    continue
            
            # Sort by similarity (highest first) and limit to top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        
        except Exception as e:
            logger.warning(f"Failed to find similar chunks: {str(e)}")
            logger.debug(f"Similarity calculation error details: {traceback.format_exc()}")
            return []

    def _cluster_by_embedding(self, df: pd.DataFrame, n_clusters: int = 10) -> List[List[str]]:
        """Cluster sections by their embeddings using KMeans."""
        from sklearn.cluster import KMeans
        
        if 'embedding' not in df.columns:
            return []
            
        # Stack embeddings into a matrix
        embeddings = np.vstack(df['embedding'].values)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(df)))
        clusters = kmeans.fit_predict(embeddings)
        
        # Group sections by cluster
        cluster_groups = [[] for _ in range(max(clusters) + 1)]
        for idx, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(df.iloc[idx]['reference'])
            
        return cluster_groups

    def _find_reference_chains(self, df: pd.DataFrame) -> List[List[str]]:
        """Find chains of connected references."""
        chains = []
        visited = set()
        
        def dfs(node: str, current_chain: List[str]):
            visited.add(node)
            current_chain.append(node)
            
            # Get references from this node
            row = df[df['reference'] == node].iloc[0]
            if row.get('reference_additional'):
                refs = row['reference_additional'].split(',')
                for ref in refs:
                    ref = ref.strip()
                    if ref and ref not in visited:
                        dfs(ref, current_chain)
        
        # Start DFS from each unvisited node
        for _, row in df.iterrows():
            if row['reference'] not in visited:
                current_chain = []
                dfs(row['reference'], current_chain)
                if len(current_chain) > 1:  # Only keep chains with multiple nodes
                    chains.append(current_chain)
        
        return chains

    def save_db(self, db_type: str = 'standard', custom_name: str = None) -> str:
        """Save the graph database to a file in the db/graph directory.
        
        Args:
            db_type: Type of graph to save ('standard' or 'hypergraph')
            custom_name: Optional custom name for the graph file
            
        Returns:
            str: Path to the saved graph file
            
        Raises:
            ValueError: If the specified graph type is not built yet
        """
        logger.debug(f"Starting to save {db_type} graph database")
        start_time = time.time()
        
        try:
            # Determine which graph to save
            if db_type == 'standard':
                graph = self.graph
            elif db_type == 'hypergraph':
                graph = self.hypergraph
            else:
                raise ValueError(f"Unknown graph type: {db_type}")
                
            # Check if graph exists
            if graph is None:
                raise ValueError(f"{db_type.capitalize()} graph not built yet")
            
            # Determine filename
            if custom_name:
                filename = f"g_{custom_name}.pkl"
            elif self.db_name:
                filename = f"g_{self.db_name}_{db_type}.pkl"
            else:
                # Generate a timestamp-based name if no db name is available
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"g_{db_type}_{timestamp}.pkl"
            
            # Create full path
            filepath = self.graph_dir / filename
            
            # Save graph using NetworkX's pickle functionality
            with open(filepath, 'wb') as f:
                pickle.dump(graph, f)
                
            logger.info(f"Saved {db_type} graph to {filepath} in {time.time() - start_time:.2f} seconds")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save {db_type} graph: {str(e)}")
            logger.debug(f"Save error details: {traceback.format_exc()}")
            raise
    
    def load_db(self, graph_file: str = None, db_type: str = 'standard') -> nx.Graph:
        """Load a graph database from a file in the db/graph directory.
        
        Args:
            graph_file: Path to the graph file (.pkl)
            db_type: Type of graph to load ('standard' or 'hypergraph')
            
        Returns:
            nx.Graph: The loaded graph
            
        Raises:
            FileNotFoundError: If the graph file doesn't exist
            ValueError: If the file format is not supported
        """
        logger.debug(f"Starting to load {db_type} graph database")
        start_time = time.time()
        
        try:
            # If no filepath is provided, try to find a graph database file
            if graph_file is None:
                # Look for files matching the pattern g_*_{db_type}.pkl
                graph_files = list(self.graph_dir.glob(f'g_*_{db_type}.pkl'))
                if not graph_files:
                    raise FileNotFoundError(f"No {db_type} graph database files found in the db/graph directory")
                # Use the most recently modified file
                graph_path = sorted(graph_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
                logger.debug(f"Using most recent {db_type} graph database file: {graph_path}")
            else:
                # Handle both relative and absolute paths
                if os.path.isabs(graph_file):
                    graph_path = Path(graph_file)
                else:
                    graph_path = self.graph_dir / graph_file
            
            # Check if file exists
            if not os.path.exists(graph_path):
                raise FileNotFoundError(f"Graph database not found: {graph_path}")
            
            # Load graph based on file extension
            if graph_path.suffix.lower() == '.pkl':
                with open(graph_path, 'rb') as f:
                    loaded_graph = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {graph_path.suffix}. Only .pkl files are supported.")
            
            # Store the loaded graph in the appropriate attribute
            if db_type == 'standard':
                self.graph = loaded_graph
                # Extract db_name from filename if possible
                filename = graph_path.stem
                if filename.startswith('g_') and '_standard' in filename:
                    self.db_name = filename[2:].replace('_standard', '')
            elif db_type == 'hypergraph':
                self.hypergraph = loaded_graph
                # Extract db_name from filename if possible
                filename = graph_path.stem
                if filename.startswith('g_') and '_hypergraph' in filename:
                    self.db_name = filename[2:].replace('_hypergraph', '')
            else:
                raise ValueError(f"Unknown graph type: {db_type}")
            
            logger.info(f"Loaded {db_type} graph from {graph_path} with {loaded_graph.number_of_nodes()} nodes and {loaded_graph.number_of_edges()} edges in {time.time() - start_time:.2f} seconds")
            return loaded_graph
            
        except Exception as e:
            logger.error(f"Failed to load graph database: {str(e)}")
            logger.debug(f"Load error details: {traceback.format_exc()}")
            raise
    
    def merge_dbs(self, graph_files: List[str], output_name: str = None, db_type: str = 'standard') -> nx.Graph:
        """Merge multiple graph databases into one.
        
        Args:
            graph_files: List of paths to graph files to merge
            output_name: Name for the merged graph file (without extension)
            db_type: Type of graphs to merge ('standard' or 'hypergraph')
            
        Returns:
            nx.Graph: The merged graph
            
        Raises:
            ValueError: If no graph files are provided or graph types are incompatible
            FileNotFoundError: If any of the specified files don't exist
        """
        logger.debug(f"Starting to merge {len(graph_files)} graph databases of type {db_type}")
        start_time = time.time()
        
        if not graph_files:
            raise ValueError("No graph files provided for merging")
            
        try:
            # Initialize merged graph based on type
            if db_type == 'standard':
                merged_graph = self.nx.MultiDiGraph()
            elif db_type == 'hypergraph':
                merged_graph = self.nx.Graph()
            else:
                raise ValueError(f"Unsupported graph type for merging: {db_type}")
            
            # Process each graph file
            processed_files = 0
            for i, graph_file in enumerate(graph_files):
                logger.debug(f"Processing graph file {i+1}/{len(graph_files)}: {graph_file}")
                
                # Handle both relative and absolute paths
                if os.path.isabs(graph_file):
                    graph_path = Path(graph_file)
                else:
                    graph_path = self.graph_dir / graph_file
                
                # Check if file exists
                if not os.path.exists(graph_path):
                    logger.warning(f"Graph file not found, skipping: {graph_path}")
                    continue
                
                # Load graph from file
                if graph_path.suffix.lower() == '.pkl':
                    with open(graph_path, 'rb') as f:
                        graph = pickle.load(f)
                else:
                    logger.warning(f"Unsupported file format, skipping: {graph_path.suffix}")
                    continue
                
                # Verify graph type
                if (db_type == 'standard' and not isinstance(graph, self.nx.MultiDiGraph)) or \
                   (db_type == 'hypergraph' and not isinstance(graph, self.nx.Graph)):
                    logger.warning(f"Graph type mismatch, skipping: {graph_path}")
                    continue
                
                # Add prefix to node IDs to avoid conflicts
                # For Chunk, Section and Document nodes, preserve the node_type but add a prefix to IDs
                node_mapping = {}
                for node, attrs in graph.nodes(data=True):
                    if 'node_type' in attrs:
                        if attrs['node_type'] == 'Chunk':
                            # Add prefix to chunk_id but preserve node ID
                            attrs['chunk_id'] = f"g{i}_{attrs['chunk_id']}"
                            node_mapping[node] = node
                        elif attrs['node_type'] == 'Document':
                            # Add prefix to document_id but preserve node ID
                            attrs['document_id'] = f"g{i}_{attrs['document_id']}"
                            node_mapping[node] = node
                        elif attrs['node_type'] == 'Section':
                            # Add prefix to section_id but preserve node ID
                            attrs['section_id'] = f"g{i}_{attrs['section_id']}"
                            node_mapping[node] = node
                        else:
                            # For other nodes, add prefix
                            node_mapping[node] = f"g{i}_{node}"
                    else:
                        # For nodes without node_type (like hyperedges), add prefix
                        node_mapping[node] = f"g{i}_{node}"
                
                # Create a copy of the graph with renamed nodes
                renamed_graph = self.nx.relabel_nodes(graph, node_mapping)
                
                # Merge into the main graph
                merged_graph.add_nodes_from(renamed_graph.nodes(data=True))
                merged_graph.add_edges_from(renamed_graph.edges(data=True))
                
                processed_files += 1
                logger.debug(f"Merged graph {i+1}: added {renamed_graph.number_of_nodes()} nodes and {renamed_graph.number_of_edges()} edges")
            
            if processed_files == 0:
                raise ValueError("No valid graph files were processed")
            
            # Generate output name if not provided
            if not output_name:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_name = f"merged_{db_type}_{timestamp}"
            
            # Remove .pkl extension if included
            if output_name.endswith('.pkl'):
                output_name = output_name[:-4]
            
            # Save the merged graph
            output_file = self.graph_dir / f"g_{output_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(merged_graph, f)
            
            # Store the merged graph in the appropriate attribute
            if db_type == 'standard':
                self.graph = merged_graph
            elif db_type == 'hypergraph':
                self.hypergraph = merged_graph
            
            # Update db_name to reflect merged status
            self.db_name = output_name
            
            logger.info(f"Merged {processed_files} graphs into {output_file} with {merged_graph.number_of_nodes()} nodes and {merged_graph.number_of_edges()} edges in {time.time() - start_time:.2f} seconds")
            return merged_graph
            
        except Exception as e:
            logger.error(f"Failed to merge graph databases: {str(e)}")
            logger.debug(f"Merge error details: {traceback.format_exc()}")
            raise
    
    def view_db(self, db_type: str = 'standard', output_path: str = None, figsize: tuple = (12, 8), seed: int = 42) -> str:
        """Generate a visual representation of the graph.
        
        Args:
            graph_type: Type of graph to visualize ('standard' or 'hypergraph')
            output_path: Path to save the visualization image. If None, will create
                         a graph_visualizations directory in the current path.
            figsize: Figure size as (width, height) in inches
            seed: Random seed for layout reproducibility
            
        Returns:
            Path to the saved visualization image
            
        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If graph_type is invalid or graph has not been built
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not installed. Please install with: pip install matplotlib")
            raise ImportError("Matplotlib is required for visualization. Install with: pip install matplotlib")
            
        # Validate graph type
        if graph_type not in ['standard', 'hypergraph']:
            raise ValueError("graph_type must be 'standard' or 'hypergraph'")
            
        # Build the graph if it hasn't been built yet
        if graph_type == 'standard':
            if self.graph is None:
                logger.info("Building standard graph first...")
                self.build_standard_graph()
            graph = self.graph
            if graph is None or len(graph.nodes) == 0:
                raise ValueError("Standard graph has not been built or has no nodes")
        else:  # hypergraph
            if self.hypergraph is None:
                logger.info("Building hypergraph first...")
                self.build_hypergraph()
            graph = self.hypergraph
            if graph is None or len(graph.nodes) == 0:
                raise ValueError("Hypergraph has not been built or has no nodes")
                
        # Set up output directory
        if output_path is None:
            viz_dir = os.path.join(os.getcwd(), "graph_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            output_path = os.path.join(viz_dir, f"{graph_type}_graph.png")
            
        # Create visualization
        logger.info(f"Generating {graph_type} graph visualization with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        plt.figure(figsize=figsize)
        pos = self.nx.spring_layout(graph, seed=seed)  # Position nodes using spring layout
        
        try:
            if graph_type == 'standard':
                # Categorize nodes by type for standard graph
                content_nodes = [n for n in graph.nodes if str(n).startswith('content_')]
                reference_nodes = [n for n in graph.nodes if isinstance(n, str) and 'Para' in str(n)]
                hierarchy_nodes = [n for n in graph.nodes if n not in content_nodes and n not in reference_nodes]
                
                # Draw nodes with different colors
                self.nx.draw_networkx_nodes(graph, pos, nodelist=content_nodes, node_color='lightblue', 
                                      node_size=500, alpha=0.8, label="Content")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=reference_nodes, node_color='lightgreen', 
                                      node_size=500, alpha=0.8, label="References")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=hierarchy_nodes, node_color='salmon', 
                                      node_size=700, alpha=0.8, label="Hierarchy")
                
                plt.title("Standard Graph Structure")
                
            else:  # hypergraph
                # Categorize nodes by type for hypergraph
                he_nodes = [n for n in graph.nodes if isinstance(n, str) and str(n).startswith('he_')]
                reference_nodes = [n for n in graph.nodes if isinstance(n, str) and 'Para' in str(n)]
                other_nodes = [n for n in graph.nodes if n not in he_nodes and n not in reference_nodes]
                
                # Draw nodes with different colors
                self.nx.draw_networkx_nodes(graph, pos, nodelist=he_nodes, node_color='purple', 
                                      node_size=700, alpha=0.8, label="Hyperedges")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=reference_nodes, node_color='lightgreen', 
                                      node_size=500, alpha=0.8, label="References")
                self.nx.draw_networkx_nodes(graph, pos, nodelist=other_nodes, node_color='orange', 
                                      node_size=500, alpha=0.8, label="Content")
                
                plt.title("Hypergraph Structure")
            
            # Draw edges and labels for both graph types
            self.nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
            
            # Draw labels with smaller font size to avoid overlap
            self.nx.draw_networkx_labels(graph, pos, font_size=8)
            
            # Add legend and finalize
            plt.legend(loc='upper right', scatterpoints=1)
            plt.axis('off')  # Turn off axis
            plt.tight_layout()
            
            # Save the visualization
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to: {output_path}")
            plt.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating graph visualization: {str(e)}")
            plt.close()
            raise