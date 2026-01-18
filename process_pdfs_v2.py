"""
PDF to Parquet Processor for RAG System

Processes PDF files and converts them to chunked Parquet format
Ready for the RAG Document Intelligence System

NEW: Automatically archives processed PDFs!

Author: Rehan Chaudhry
"""

import PyPDF2
import pandas as pd
import re
from pathlib import Path
from datetime import datetime

class PDFProcessor:
    """Process PDFs and create Parquet files for RAG"""
    
    def __init__(self, chunk_size=500, overlap=50):
        """
        Initialize processor
        
        Args:
            chunk_size: Words per chunk (default: 500)
            overlap: Overlapping words between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract all text from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            str: Extracted text
        """
        print(f"📖 Extracting text from: {pdf_path}")
        
        text = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            print(f"   Pages: {num_pages}")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        # Clean text
        text = self.clean_text(text)
        
        print(f"   Extracted: {len(text)} characters, {len(text.split())} words")
        
        return text
    
    def clean_text(self, text):
        """
        Clean extracted text
        
        Args:
            text: Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text):
        """
        Split text into overlapping chunks
        
        Args:
            text: Full text to chunk
            
        Returns:
            list: List of text chunks
        """
        words = text.split()
        chunks = []
        
        i = 0
        chunk_num = 0
        
        while i < len(words):
            # Get chunk_size words
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'chunk_id': f'chunk_{chunk_num}',
                'text': chunk_text,
                'chunk_index': chunk_num,
                'word_count': len(chunk_words)
            })
            
            # Move forward by (chunk_size - overlap)
            i += (self.chunk_size - self.overlap)
            chunk_num += 1
        
        print(f"   Created: {len(chunks)} chunks")
        
        return chunks
    
    def process_pdf(self, pdf_path, output_path=None):
        """
        Complete processing: PDF → Parquet
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output Parquet (optional)
            
        Returns:
            dict: Processing results
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING: {Path(pdf_path).name}")
        print(f"{'='*60}")
        
        # Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Create chunks
        chunks = self.chunk_text(full_text)
        
        # Convert to DataFrame
        df = pd.DataFrame(chunks)
        
        # Add source filename
        df['source_file'] = Path(pdf_path).name
        
        # Determine output path
        if output_path is None:
            output_name = Path(pdf_path).stem + '_chunks.parquet'
            output_path = Path('data/processed') / output_name
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Parquet
        df.to_parquet(output_path, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"   Output: {output_path}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Total words: {df['word_count'].sum()}")
        print(f"{'='*60}\n")
        
        return {
            'pdf_path': str(pdf_path),
            'output_path': str(output_path),
            'num_chunks': len(chunks),
            'total_words': df['word_count'].sum()
        }


def process_all_pdfs(pdf_dir='data/pdfs', output_dir='data/processed', archive_dir='data/archive'):
    """
    Process all PDFs in a directory and archive them
    
    Features:
    - Checks if pdf_dir is empty → does nothing if empty
    - Processes all PDFs found
    - Moves processed PDFs to archive_dir
    - Keeps pdf_dir clean for next batch
    
    Args:
        pdf_dir: Directory containing PDFs to process
        output_dir: Directory for output Parquet files
        archive_dir: Directory to move processed PDFs
        
    Returns:
        list: Processing results for each PDF
    """
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    archive_dir = Path(archive_dir)
    
    # Ensure directories exist
    pdf_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(pdf_dir.glob('*.pdf'))
    
    # If no PDFs, do nothing
    if not pdf_files:
        print(f"\n📁 Checking: {pdf_dir}")
        print(f"✅ No PDF files found - nothing to process")
        print(f"\n💡 TIP: Place PDF files in '{pdf_dir}' to process them automatically")
        return []
    
    # PDFs found - start processing
    print(f"\n{'='*60}")
    print(f"FOUND {len(pdf_files)} PDF FILE(S) TO PROCESS")
    print(f"{'='*60}")
    
    for pdf_file in pdf_files:
        print(f"📄 {pdf_file.name}")
    
    print(f"\n{'='*60}")
    print("STARTING BATCH PROCESSING")
    print(f"{'='*60}\n")
    
    processor = PDFProcessor(chunk_size=500, overlap=50)
    results = []
    
    for pdf_file in pdf_files:
        try:
            # Process PDF → Parquet
            result = processor.process_pdf(pdf_file)
            results.append(result)
            
            # Move PDF to archive
            archive_path = archive_dir / pdf_file.name
            
            # If file already exists in archive, add timestamp
            if archive_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_path = archive_dir / f"{pdf_file.stem}_{timestamp}{pdf_file.suffix}"
            
            pdf_file.rename(archive_path)
            print(f"📦 Archived: {pdf_file.name} → {archive_dir.name}/")
            
        except Exception as e:
            print(f"\n❌ ERROR processing {pdf_file.name}: {str(e)}")
            print(f"   File NOT archived (still in {pdf_dir.name}/)")
            print(f"   Fix the error and run again\n")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(results)}/{len(pdf_files)} file(s)")
    
    if results:
        print(f"📊 Total chunks created: {sum(r['num_chunks'] for r in results)}")
        print(f"📝 Total words processed: {sum(r['total_words'] for r in results):,}")
        print(f"📦 Processed PDFs moved to: {archive_dir}/")
        print(f"📁 {pdf_dir}/ is now empty and ready for new PDFs")
    
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    """
    Run this script to process all PDFs in data/pdfs/
    
    Usage:
        python process_pdfs.py
        
    Workflow:
        1. Place PDFs in data/pdfs/
        2. Run: python process_pdfs.py
        3. Parquet files created in data/processed/
        4. PDFs moved to data/archive/
        5. data/pdfs/ is empty, ready for next batch!
    """
    
    print("\n🚀 PDF TO PARQUET PROCESSOR")
    print("="*60)
    
    # Process all PDFs in data/pdfs folder
    results = process_all_pdfs()
    
    if results:
        print("\n📋 OUTPUT FILES CREATED:")
        for result in results:
            print(f"   ✅ {result['output_path']}")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Update config.py PARQUET_FILES with new files")
        print("   2. Update config.py DOCUMENT_METADATA with paper details")
        print("   3. Delete old embeddings cache: rm cache/embeddings.npy")
        print("   4. Rebuild embeddings: python embedding_engine.py")
        print("   5. Test RAG system: streamlit run streamlit_app.py")
        
        print("\n📁 FOLDER STATUS:")
        print(f"   data/pdfs/      → Empty (ready for new PDFs)")
        print(f"   data/processed/ → {len(results)} new Parquet file(s)")
        print(f"   data/archive/   → {len(results)} archived PDF(s)")
    
    else:
        print("\n📁 FOLDER STATUS:")
        print(f"   data/pdfs/ is empty")
        print(f"   No processing needed")
        print(f"\n   Place PDF files in data/pdfs/ and run again!")
