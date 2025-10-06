
import pandas as pd
import os
from sodapy import Socrata
import pathlib
import requests
from zipfile import ZipFile
from tqdm.auto import tqdm
import pathlib
from langchain_huggingface import HuggingFaceEmbeddings as lgHuggingFaceEmbeddings
from tqdm import tqdm
from langchain_community.document_loaders import BSHTMLLoader


def get_data_directory() -> pathlib.Path:
    
    # Use current working directory where the user is running the package
    user_cwd = pathlib.Path.cwd()
    data_dir = user_cwd / "cdc-corpus-data"
    
    return data_dir


def download_file(url: str, file_name: str) -> str:
    """Download a file with rich progress bar."""
    
    # Create the directory if it doesn't exist
    os.makedirs(get_data_directory() / "zip", exist_ok=True)
    local_filename = get_data_directory() / "zip"/ file_name
 
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                if chunk:
                    size = f.write(chunk)
                    #progress.update(task, advance=size)
    
    return local_filename


def extract_zip_files() -> None:
        """Extract all collection zip files to the json-html directory."""

        data_dir = get_data_directory()
        zip_dir = data_dir  / "zip"
        html_dir = data_dir / "html-outputs"
        
        # Create the directory if it doesn't exist
        os.makedirs(html_dir, exist_ok=True)
        
        if not data_dir.exists():
            print(f"Zip directory {zip_dir} does not exist")
            return
        
        for zip_file in os.listdir(zip_dir):
            if zip_file.endswith('.zip'):
                with ZipFile(zip_dir / zip_file, 'r') as zip_obj:
                    zip_obj.extractall(html_dir)

def _load_html_file(file_path_name: str) -> str:
        """Load HTML content for the specified paper.
        
        Tries multiple encodings to handle different file formats:
        1. UTF-8 (most common)
        2. unicode_escape (fallback for problematic files)
        3. latin-1 (final fallback)
        """
        encodings_to_try = ['utf-8', 'unicode_escape', 'latin-1']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path_name, encoding=encoding) as f:
                    html_content = f.read()
                
                loader = BSHTMLLoader(
                    file_path=file_path_name,
                    open_encoding=encoding)
                docs = []
                docs_lazy = loader.lazy_load()
                for doc in docs_lazy:
                    docs.append(doc)
                return html_content, docs[0].metadata
            except UnicodeDecodeError:
                # Try next encoding
                continue
            except FileNotFoundError:
                print(f"File not found: {file_path_name}")
                raise FileNotFoundError(f'Unable to find HTML file: {file_path_name}')
            except Exception as e:
                # For other errors, try next encoding
                print(f"Error reading {file_path_name} with {encoding}: {e}")
                continue
        
        # If all encodings failed
        raise UnicodeDecodeError(
            'multiple_encodings', 
            b'', 0, 0, 
            f'Unable to decode {file_path_name} with any of the tried encodings: {encodings_to_try}'
        )


def _load_bs_html(file_path_name: str) -> str:

    encodings_to_try = ['utf-8', 'unicode_escape', 'latin-1']

    for encoding in encodings_to_try:
        try :
            loader = BSHTMLLoader(
                file_path=file_path_name,
                open_encoding=encoding
            )
            docs = []
            docs_lazy = loader.lazy_load()

            # async variant:
            # docs_lazy = await loader.alazy_load()
            #print(len(list(docs_lazy)))
            for doc in docs_lazy:
                docs.append(doc)
            #print(docs[0].metadata)
            #print(docs[0].page_content)
            return docs[0].page_content,docs[0].metadata
        except UnicodeDecodeError:
                # Try next encoding
                continue
        except FileNotFoundError:
                print(f"File not found: {file_path_name}")
                raise FileNotFoundError(f'Unable to find HTML file: {file_path_name}')
        except Exception as e:
                # For other errors, try next encoding
                print(f"Error reading {file_path_name} with {encoding}: {e}")
                continue
        
        # If all encodings failed
        raise UnicodeDecodeError(
            'multiple_encodings', 
            b'', 0, 0, 
            f'Unable to decode {file_path_name} with any of the tried encodings: {encodings_to_try}'
        )

def load_html_files():    
    data_dir = get_data_directory()
    collection_base_dir = data_dir / "html-outputs" / "pcd"
    articles_html = {}
    folder_to_scan = "issues"
    target_dir = collection_base_dir / folder_to_scan
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.htm') or file.endswith('.html'):
                file_path = pathlib.Path(root) / file
                relative_path = file_path.relative_to(collection_base_dir)
            
                #Filter out non-content files
                file_lower = file.lower()
                if ('cover' in file_lower or 
                   'ac-' in file_lower or
                   'toc' in file_lower or
                   'index' in file_lower or
                   'archive' in file_lower):
                   continue
                # filter Erratum
                if file.endswith('e.htm'):
                    continue

                # Skip non-English files (those ending with language codes)
                if file.endswith(('_es.htm', '_fr.htm', '_zhs.htm', '_zht.htm')):
                    continue
                try:
                    # Use the existing _load_html_file method for consistency
                    html_content,metadata = _load_html_file(str(file_path))
                    #html_content,metadata = _load_bs_html(str(file_path))
                    # Use the relative path as the key to maintain structure
                    articles_html[str(relative_path)] = {"html_content":html_content,
                                                         "metadata":metadata}
              
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                
            
    print(f"Loaded {len(articles_html)} HTML articles")
    return articles_html


if __name__=="__main__":
    data = load_html_files()
    meta =[]
    for k,v in data.items():
        meta.append(v['metadata'])

    df = pd.DataFrame(meta)
    print(df['title'])
    df.to_csv("data.csv")