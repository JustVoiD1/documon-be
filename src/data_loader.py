import os
import tempfile
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_path = Path(data_dir).resolve()
        self.pdf_files = list(self.data_path.glob('**/*.pdf'))
        self.txt_files = list(self.data_path.glob('**/*.txt'))
        self.csv_files = list(self.data_path.glob('**/*.csv'))
        self.xlsx_files = list(self.data_path.glob('**/*.xlsx'))
        self.docx_files = list(self.data_path.glob('**/*.docx'))
        self.json_files = list(self.data_path.glob('**/*.json'))

        self.documents = []

    def load_pdf_files(self):
        print(f"[DEBUG] Found {len(self.pdf_files)} PDF files: {[str(f) for f in self.pdf_files]}")
        for pdf_file in self.pdf_files:
            print(f"[DEBUG] Loading PDF: {pdf_file}")
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
                self.documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")
    
    def load_txt_files(self):
        print(f"[DEBUG] Found {len(self.txt_files)} TXT files: {[str(f) for f in self.txt_files]}")
        for txt_file in self.txt_files:
            print(f"[DEBUG] Loading TXT: {txt_file}")
            try:
                loader = TextLoader(str(txt_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
                self.documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load TXT {txt_file}: {e}")
    
    
    def load_csv_files(self):
        print(f"[DEBUG] Found {len(self.csv_files)} CSV files: {[str(f) for f in self.csv_files]}")
        for csv_file in self.csv_files:
            print(f"[DEBUG] Loading CSV: {csv_file}")
            try:
                loader = CSVLoader(str(csv_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
                self.documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load CSV {csv_file}: {e}")
    
    def load_excel_files(self):
        print(f"[DEBUG] Found {len(self.xlsx_files)} Excel files: {[str(f) for f in self.xlsx_files]}")
        for xlsx_file in self.xlsx_files:
            print(f"[DEBUG] Loading Excel: {xlsx_file}")
            try:
                loader = UnstructuredExcelLoader(str(xlsx_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
                self.documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    def load_word_files(self):
        print(f"[DEBUG] Found {len(self.docx_files)} Word files: {[str(f) for f in self.docx_files]}")
        for docx_file in self.docx_files:
            print(f"[DEBUG] Loading Word: {docx_file}")
            try:
                loader = UnstructuredWordDocumentLoader(str(docx_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
                self.documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    def load_json_files(self):
        print(f"[DEBUG] Found {len(self.json_files)} JSON files: {[str(f) for f in self.json_files]}")
        for json_file in self.json_files:
            print(f"[DEBUG] Loading JSON: {json_file}")
            try:
                loader = JSONLoader(file_path=str(json_file), jq_schema=".", text_content=False)
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
                self.documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    

    def load_single_file(self, file_path: Path) -> List[Document]:
        """
        Load a single file by path and return its LangChain Document objects.
        """
        file_path = Path(file_path).resolve()
        ext = file_path.suffix.lower()
        file_str = str(file_path)

        print(f"[DEBUG] Loading file: {file_path}")
        try:
            if ext == ".pdf":
                return PyMuPDFLoader(file_str).load()
            elif ext == ".txt":
                return TextLoader(file_str).load()
            elif ext == ".csv":
                return CSVLoader(file_str).load()
            elif ext in [".xlsx", ".xls"]:
                return UnstructuredExcelLoader(file_str).load()
            elif ext in [".docx", ".doc"]:
                return UnstructuredWordDocumentLoader(file_str).load()
            elif ext == ".json":
                return JSONLoader(file_path=file_str, jq_schema=".", text_content=False).load()
            else:
                print(f"[WARN] Unsupported file extension: {ext}")
                return []
        except Exception as e:
            print(f"[ERROR] Failed to load file {file_path}: {e}")
            return []

    def load_from_bytes(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Parses document content directly from bytes via temporary execution without saving
        any persistent local files on disk. The temporary file is immediately purged.
        """
        ext = Path(filename).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            docs = self.load_single_file(tmp_path)
            for doc in docs:
                if isinstance(doc.metadata, dict) and "source" in doc.metadata:
                    doc.metadata["source"] = filename
            return docs
        finally:
            if tmp_path.exists():
                try:
                    os.remove(tmp_path)
                except Exception as cleanup_err:
                    print(f"[WARN] Temporary file cleanup warning: {cleanup_err}")


    def load_all_documents(self) -> List[Document]:
        """
        Load all supported files from the data directory and convert to LangChain document structure.
        Supported: PDF, TXT, CSV, Excel, Word, JSON
        """
        # Use project root data folder
        print(f"[DEBUG] Data path: {self.data_path}")

        # PDF files
        self.load_pdf_files()

        # TXT files
        self.load_txt_files()

        # CSV files
        self.load_csv_files()

        # Excel files
        self.load_excel_files()

        # Word files
        self.load_word_files()

        # JSON files
        self.load_json_files()

        print(f"[DEBUG] Total loaded documents: {len(self.documents)}")
        return self.documents

# Example usage
# if __name__ == "__main__":
#     docs = load_all_documents("data")
#     print(f"Loaded {len(docs)} documents.")
#     print("Example document:", docs[0] if docs else None)