import pdfplumber
import pandas as pd
import fitz  # PyMuPDF
import docx
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from django.conf import settings
import os
import mimetypes
from dotenv import load_dotenv
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import re

load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-pro')

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
try:
    # Ensure ChromaDB directory exists
    os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
    print(f"DEBUG: ChromaDB directory: {settings.CHROMA_DB_DIR}")
    
    # Configure ChromaDB settings
    chroma_settings = chromadb.config.Settings(
        is_persistent=True,
        persist_directory=settings.CHROMA_DB_DIR,
        allow_reset=True,
        anonymized_telemetry=False
    )
    
    chroma_client = chromadb.PersistentClient(
        path=settings.CHROMA_DB_DIR,
        settings=chroma_settings
    )
    print("DEBUG: ChromaDB client initialized")
    collection = None
except Exception as e:
    print(f"ERROR initializing ChromaDB client: {str(e)}")
    raise e

def get_or_create_collection():
    """Get the existing collection or create a new one if it doesn't exist."""
    global collection, chroma_client
    try:
        # Ensure client exists
        if chroma_client is None:
            print("DEBUG: Reinitializing ChromaDB client")
            os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
            chroma_settings = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=settings.CHROMA_DB_DIR,
                allow_reset=True,
                anonymized_telemetry=False
            )
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_DIR,
                settings=chroma_settings
            )
        
        # Try to get the existing collection
        try:
            collection = chroma_client.get_collection(name="document_chunks")
            print("DEBUG: Retrieved existing collection")
        except Exception as e:
            print(f"DEBUG: Collection not found, creating new one: {str(e)}")
            # Create a new collection if it doesn't exist
            collection = chroma_client.create_collection(
                name="document_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            print("DEBUG: Created new collection")
        
        return collection
    except Exception as e:
        print(f"ERROR in get_or_create_collection: {str(e)}")
        raise e

# Initialize collection on module load
try:
    collection = get_or_create_collection()
    print("DEBUG: Initial collection setup complete")
except Exception as e:
    print(f"ERROR during initial collection setup: {str(e)}")
    raise e

def init_chroma():
    """Initialize or reinitialize ChromaDB collection."""
    global collection, chroma_client
    try:
        # Ensure client exists
        if chroma_client is None:
            os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
            chroma_settings = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=settings.CHROMA_DB_DIR,
                allow_reset=True,
                anonymized_telemetry=False
            )
            chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_DIR,
                settings=chroma_settings
            )
        
        # Force delete the existing collection if it exists
        try:
            chroma_client.delete_collection("document_chunks")
            print("DEBUG: Deleted existing collection")
        except Exception as e:
            print(f"DEBUG: No existing collection to delete: {str(e)}")
        
        # Create a new collection
        collection = chroma_client.create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        print("DEBUG: Created new collection")
        return collection
    except Exception as e:
        print(f"ERROR initializing ChromaDB: {str(e)}")
        raise e

def list_all_chunks():
    """List all chunks in ChromaDB for debugging."""
    try:
        results = collection.get()
        print("\nDEBUG: All chunks in ChromaDB:")
        print(f"Total chunks: {len(results['ids'])}")
        for id, meta in zip(results['ids'], results['metadatas']):
            print(f"Chunk ID: {id}, Document ID: {meta.get('document_id')}")
        print("\n")
    except Exception as e:
        print(f"Error listing chunks: {str(e)}")

def reset_chroma_collection():
    """Reset the ChromaDB collection by deleting and recreating it."""
    global collection
    try:
        chroma_client.delete_collection("document_chunks")
    except:
        pass
    collection = chroma_client.create_collection("document_chunks")

def detect_file_type(file_path: str) -> str:
    """Detect file type using file extension and mime types."""
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    if mime_type == 'application/pdf' or extension == '.pdf':
        return 'pdf'
    elif mime_type == 'text/csv' or extension == '.csv':
        return 'csv'
    elif mime_type == 'text/plain' or extension == '.txt':
        return 'txt'
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or extension == '.docx':
        return 'docx'
    elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or extension == '.xlsx':
        return 'xlsx'
    elif mime_type and mime_type.startswith('image/'):
        return 'image'
    else:
        raise ValueError(f"Unsupported file type: {mime_type or extension}")

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Process PDF file and return chunks with page numbers."""
    chunks = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                chunks.append({
                    'content': text,
                    'page_number': page_num,
                    'metadata': {
                        'file_type': 'pdf',
                        'page': page_num
                    }
                })
    
    return chunks

def process_csv(file_path: str) -> List[Dict[str, Any]]:
    """Process CSV file and return chunks with table information."""
    df = pd.read_csv(file_path)
    chunks = []
    
    # Convert DataFrame to string representation
    table_str = df.to_string()
    chunks.append({
        'content': table_str,
        'metadata': {
            'file_type': 'csv',
            'columns': df.columns.tolist(),
            'rows': len(df)
        }
    })
    
    return chunks

def process_txt(file_path: str) -> List[Dict[str, Any]]:
    """Process text file and return chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into paragraphs
    paragraphs = content.split('\n\n')
    chunks = []
    
    for i, para in enumerate(paragraphs):
        if para.strip():
            chunks.append({
                'content': para.strip(),
                'metadata': {
                    'file_type': 'txt',
                    'paragraph': i + 1
                }
            })
    
    return chunks

def process_docx(file_path: str) -> List[Dict[str, Any]]:
    """Process Word document and return chunks."""
    doc = docx.Document(file_path)
    chunks = []
    
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            chunks.append({
                'content': para.text.strip(),
                'metadata': {
                    'file_type': 'docx',
                    'paragraph': i + 1
                }
            })
    
    return chunks

def process_image(file_path: str) -> List[Dict[str, Any]]:
    """Process image file using Gemini Vision."""
    image = Image.open(file_path)
    
    # Use Gemini Vision to analyze the image
    response = model.generate_content([
        "Analyze this image and provide a detailed description of its contents.",
        image
    ])
    
    chunks = [{
        'content': response.text,
        'metadata': {
            'file_type': 'image',
            'format': image.format,
            'size': image.size
        }
    }]
    
    return chunks

def process_excel(file_path: str) -> List[Dict[str, Any]]:
    """Process Excel file and return chunks with structured data."""
    chunks = []
    
    # Read all sheets
    excel_file = pd.ExcelFile(file_path)
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Convert DataFrame to JSON-serializable format
        df_dict = df.copy()
        
        # Convert timestamps to ISO format strings
        for col in df_dict.select_dtypes(include=['datetime64[ns]']).columns:
            df_dict[col] = df_dict[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
        # Convert numeric types to Python native types
        for col in df_dict.select_dtypes(include=[np.number]).columns:
            df_dict[col] = df_dict[col].astype(float).tolist()
            
        # Store the raw data as JSON string
        raw_data = {
            'data': df_dict.to_dict(orient='records'),
            'columns': df_dict.columns.tolist(),
            'sheet_name': sheet_name
        }
        
        # Create a text description that includes both summary and actual data
        description = f"Sheet: {sheet_name}\n"
        description += f"Columns: {', '.join(df.columns)}\n"
        description += f"Number of rows: {len(df)}\n\n"
        
        # Add actual data rows
        description += "Data rows:\n"
        for idx, row in df.iterrows():
            description += "* "
            for col in df.columns:
                description += f"{col}: {row[col]}, "
            description = description.rstrip(", ") + "\n"
        
        # Add summary statistics for numeric columns
        description += "\nNumeric column summaries:\n"
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                description += f"{col}:\n"
                description += f"  Min: {df[col].min()}\n"
                description += f"  Max: {df[col].max()}\n"
                description += f"  Mean: {df[col].mean():.2f}\n"
        
        # Store both raw data and description
        content = {
            'description': description,
            'raw_data': raw_data
        }
        
        chunks.append({
            'content': json.dumps(content),  # Store as JSON string
            'metadata': {
                'file_type': 'xlsx',
                'sheet_name': sheet_name
            }
        })
    
    return chunks

def get_embeddings(text: str) -> List[float]:
    """Generate embeddings for text using sentence-transformers."""
    return embedding_model.encode(text).tolist()

def delete_document_chunks(document_id: str):
    """Delete all chunks for a document from ChromaDB."""
    try:
        # Get all chunks for the document
        results = collection.get(
            where={"document_id": document_id}
        )
        if results and results['ids']:
            collection.delete(ids=results['ids'])
    except Exception as e:
        print(f"Error deleting document chunks: {str(e)}")

def store_chunks(chunks: List[Dict[str, Any]], document_id: str):
    """Store chunks in ChromaDB with embeddings."""
    global collection
    try:
        print(f"\nDEBUG: Storing chunks for document {document_id}")
        
        # Ensure we have a valid collection
        if collection is None:
            print("DEBUG: Collection is None, attempting to get or create")
            collection = get_or_create_collection()
        
        try:
            # Verify collection exists by attempting a simple operation
            count = collection.count()
            print(f"DEBUG: Collection verified, current count: {count}")
        except Exception as e:
            print(f"DEBUG: Collection verification failed: {str(e)}, recreating collection")
            collection = get_or_create_collection()
        
        try:
            # List current chunks
            results = collection.get()
            print(f"DEBUG: Current chunk count: {len(results['ids']) if 'ids' in results else 0}")
        except Exception as e:
            print(f"DEBUG: Error listing chunks (non-fatal): {str(e)}")
        
        # Delete existing chunks for this document
        try:
            results = collection.get(
                where={"document_id": document_id}
            )
            if results and results['ids']:
                print(f"DEBUG: Deleting {len(results['ids'])} existing chunks for document {document_id}")
                collection.delete(ids=results['ids'])
        except Exception as e:
            print(f"DEBUG: Error deleting existing chunks (non-fatal): {str(e)}")
        
        # Store new chunks
        for i, chunk in enumerate(chunks):
            try:
                chunk_id = f"{document_id}_{i}"
                embedding = get_embeddings(chunk['content'])
                metadata = {
                    'document_id': document_id,
                    'chunk_index': i,
                    **chunk['metadata']
                }
                
                print(f"DEBUG: Adding chunk {i} with ID {chunk_id}")
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk['content']],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                print(f"DEBUG: Successfully stored chunk {i}")
            except Exception as e:
                print(f"ERROR storing chunk {i}: {str(e)}")
                # Continue with next chunk instead of failing completely
                continue
        
        # Verify final state
        try:
            final_count = collection.count()
            print(f"DEBUG: Final collection count: {final_count}")
        except Exception as e:
            print(f"DEBUG: Error getting final count: {str(e)}")
        
    except Exception as e:
        print(f"ERROR in store_chunks: {str(e)}")
        # Try to recover the collection but don't retry the operation
        try:
            collection = get_or_create_collection()
        except:
            pass
        raise e

def query_chunks(query: str, n_results: int = 5, document_ids: List[str] = None) -> List[Dict[str, Any]]:
    """Query ChromaDB for relevant chunks."""
    global collection
    print(f"\nDEBUG: Querying chunks with: {query}")
    if document_ids:
        print(f"DEBUG: Filtering to document IDs: {document_ids}")
    
    try:
        # Ensure we have a valid collection
        if collection is None:
            print("DEBUG: Collection is None, attempting to get or create")
            collection = get_or_create_collection()
        
        try:
            # Verify collection exists by attempting a simple operation
            count = collection.count()
            print(f"DEBUG: Collection verified, current count: {count}")
        except Exception as e:
            print(f"DEBUG: Collection verification failed: {str(e)}, recreating collection")
            collection = get_or_create_collection()
        
        query_embedding = get_embeddings(query)
        print("DEBUG: Generated query embeddings")
        
        # Define query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ['documents', 'metadatas']
        }
        
        # Add document_id filter if provided
        if document_ids:
            query_params["where"] = {"document_id": {"$in": document_ids}}
        
        results = collection.query(**query_params)
        
        if not results['documents'] or not results['documents'][0]:
            print("DEBUG: No results found")
            return []
        
        print(f"DEBUG: Query returned {len(results['documents'][0])} results")
        
        # Debug print for document IDs in results
        for meta in results['metadatas'][0]:
            print(f"DEBUG: Found chunk with document_id: {meta.get('document_id')}")
        
        return [
            {
                'content': doc,
                'metadata': meta
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    except Exception as e:
        print(f"ERROR in query_chunks: {str(e)}")
        # Try to recover the collection but return empty results
        try:
            collection = get_or_create_collection()
        except:
            pass
        return []

def detect_visualization_type(response_text: str, data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Detect appropriate visualization type based on response text, data, and metadata."""
    response_lower = response_text.lower()
    
    # First check for trend keyword - this takes highest priority
    if 'trend' in response_lower:
        return 'line'
    
    # Keywords that suggest different chart types
    pie_keywords = ['breakdown', 'distribution', 'composition', 'ratio', 'percentage', 'proportion', 'share', 'split', 'pie']
    time_keywords = ['over time', 'timeline', 'historical', 'monthly', 'yearly', 'daily', 'weekly', 'time series']
    comparison_keywords = ['compare', 'comparison', 'difference', 'ranking', 'rank', 'versus', 'vs']
    
    # Check if data looks like time series
    x_values = data['x_values']
    is_time_series = False
    try:
        # Try to parse x values as dates
        pd.to_datetime(x_values)
        is_time_series = True
    except:
        # Check if x values are months
        months = ['january', 'february', 'march', 'april', 'may', 'june', 
                 'july', 'august', 'september', 'october', 'november', 'december']
        if all(str(x).lower().strip() in months for x in x_values):
            is_time_series = True
    
    # Check if data looks like percentages or parts of a whole
    y_values = data['y_values']
    total = sum(y_values)
    looks_like_percentage = (
        len(x_values) <= 10 and  # Not too many categories
        (metadata.get('unit') == '%' or 
         any('percent' in str(x).lower() for x in x_values) or
         (90 <= total <= 110))  # Sum close to 100
    )
    
    # Decision logic
    if any(keyword in response_lower for keyword in pie_keywords) or looks_like_percentage:
        return 'pie'
    elif is_time_series or any(keyword in response_lower for keyword in time_keywords):
        return 'line'
    elif any(keyword in response_lower for keyword in comparison_keywords) or len(x_values) > 1:
        return 'bar'
    
    # Default to bar for comparison if no other type is clearly indicated
    return 'bar'

def is_datetime_column(series: pd.Series) -> bool:
    """Check if a column contains datetime values."""
    return pd.api.types.is_datetime64_any_dtype(series) or (
        series.dtype == object and 
        all(isinstance(x, (datetime, str)) and is_date_string(str(x)) for x in series.dropna())
    )

def is_date_string(s: str) -> bool:
    """Check if a string represents a date."""
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
    ]
    return any(re.match(pattern, s) for pattern in date_patterns)

def create_visualization(data: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    """Create appropriate visualization based on the data and query."""
    try:
        print("\nDEBUG: Creating visualization")
        print(f"DEBUG: Data structure: {data.keys()}")
        
        # Create DataFrame from the data
        df = pd.DataFrame(data['data'])
        print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: DataFrame head:\n{df.head()}")
        
        # Clean up the data
        # Convert string numbers to float
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Remove currency symbols and commas
                    cleaned = df[col].str.replace('$', '').str.replace(',', '')
                    df[col] = pd.to_numeric(cleaned, errors='ignore')
                except:
                    pass
        
        # Try to identify key columns for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # For time series data, try to parse dates
        for col in categorical_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except:
                pass
        
        # Detect visualization type
        viz_type = detect_visualization_type(query, df, {})
        print(f"DEBUG: Detected visualization type: {viz_type}")
        
        if viz_type:
            if viz_type == 'pie':
                fig = create_pie_chart(df, query)
            elif viz_type == 'bar':
                fig = create_bar_chart(df, query)
            elif viz_type == 'line':
                fig = create_line_chart(df, query)
            
            # Convert to HTML
            chart_html = fig.to_html(full_html=False, include_plotlyjs=True)
            print("DEBUG: Successfully created visualization HTML")
            
            return {
                'type': 'visualization',
                'content': chart_html
            }
        
        return None
    except Exception as e:
        print(f"DEBUG: Error creating visualization: {str(e)}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return None

def create_pie_chart(df: pd.DataFrame, query: str) -> go.Figure:
    """Create a pie chart based on the data and query."""
    # Find numeric columns for values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for pie chart")
    
    # Try to find the most relevant numeric column based on the query
    value_keywords = ['value', 'amount', 'revenue', 'sales', 'units', 'quantity', 'percentage', 'percent', 'share']
    values_col = None
    for keyword in value_keywords:
        matching_cols = [col for col in numeric_cols if keyword.lower() in col.lower()]
        if matching_cols:
            values_col = matching_cols[0]
            break
    
    # If no specific column found, use the first numeric column
    if not values_col:
        values_col = numeric_cols[0]
    
    # Find a suitable column for labels (prefer non-numeric)
    label_cols = [col for col in df.columns if col not in numeric_cols]
    labels_col = label_cols[0] if label_cols else df.index
    
    # Determine if values are percentages
    is_percentage = False
    sum_values = df[values_col].sum()
    if 90 <= sum_values <= 110:  # Sum close to 100%
        is_percentage = True
    
    # Create the pie chart
    fig = px.pie(
        df,
        values=values_col,
        names=labels_col,
        title=f"Distribution of {values_col} by {labels_col}"
    )
    
    # Update layout for better appearance
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        hole=0.3,  # Create a donut chart for better visual appearance
        marker=dict(line=dict(color='#FFFFFF', width=1))
    )
    
    # Add percentage symbol to hover template if needed
    if is_percentage:
        fig.update_traces(
            hovertemplate='%{label}<br>%{percent}<br>Value: %{value}%'
        )
    
    # Improve layout
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_bar_chart(df: pd.DataFrame, query: str) -> go.Figure:
    """Create a bar chart based on the data and query."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for bar chart")
    
    # Use the first numeric column as y-axis
    y_col = numeric_cols[0]
    
    # Find a suitable column for x-axis (prefer non-numeric)
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    x_col = non_numeric_cols[0] if non_numeric_cols else df.index
    
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=f"{y_col} by {x_col}"
    )
    return fig

def create_line_chart(df: pd.DataFrame, query: str) -> go.Figure:
    """Create a line chart based on the data and query."""
    # Find datetime column
    date_col = None
    for col in df.columns:
        if is_datetime_column(df[col]):
            date_col = col
            break
    
    if not date_col:
        raise ValueError("No datetime column found for line chart")
    
    # Find numeric column for y-axis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for line chart")
    
    y_col = numeric_cols[0]
    
    # Sort by date
    df = df.sort_values(date_col)
    
    fig = px.line(
        df,
        x=date_col,
        y=y_col,
        title=f"{y_col} over time"
    )
    return fig

def extract_data_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract data points from Gemini's response text."""
    try:
        print("DEBUG: Attempting to extract data from response")
        
        # Initialize data containers
        x_values = []
        y_values = []
        x_label = None
        y_label = None
        unit = None
        
        # Try to identify the data type and unit from the response
        response_lower = response_text.lower()
        
        # Look for common units and data types
        if "%" in response_text:
            unit = "%"
        elif "°c" in response_text.lower() or "degrees" in response_text.lower():
            unit = "°C"
        elif "mm" in response_text.lower() or "rainfall" in response_text.lower():
            unit = "mm"
        elif "$" in response_text or "usd" in response_text.lower():
            unit = "$"
            
        # Look for data rows marked with asterisks or bullet points
        lines = response_text.split('\n')
        data_row_pattern = r'\*\s*(.*?):\s*([\d,.]+)'
        
        for line in lines:
            line = line.strip()
            
            # Try to match the data row pattern
            match = re.search(data_row_pattern, line)
            if match:
                x_val = match.group(1).strip()
                value_str = match.group(2).strip()
                
                # Clean up the value string
                value_str = value_str.replace(',', '').replace('$', '').replace('%', '')
                try:
                    y_val = float(value_str)
                    x_values.append(x_val)
                    y_values.append(y_val)
                except ValueError:
                    continue
            
            # Also look for "key: value" pairs in the data rows
            elif ': ' in line and not line.startswith('Source:') and not line.startswith('Content:'):
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    key = parts[0].strip('* ')
                    value_str = parts[1].strip()
                    
                    # Try to extract numeric value
                    value_str = ''.join(c for c in value_str if c.isdigit() or c in '.-')
                    try:
                        value = float(value_str)
                        x_values.append(key)
                        y_values.append(value)
                    except ValueError:
                        continue
        
        if x_values and y_values:
            print(f"DEBUG: Extracted {len(x_values)} data points")
            
            # Try to determine labels from the data
            if not x_label:
                if all(x.lower().strip() in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'] for x in x_values):
                    x_label = "Month"
                elif all(str(x).isdigit() for x in x_values):
                    x_label = "Year"
                else:
                    # Try to find a common prefix in x values
                    common_words = set(x_values[0].split()[0].lower())
                    for x in x_values[1:]:
                        if not x.split():
                            continue
                        common_words &= set(x.split()[0].lower())
                    if common_words:
                        x_label = next(iter(common_words)).title()
                    else:
                        x_label = "Category"
            
            # If y_label is still not set, try to determine from the response
            if not y_label:
                value_keywords = ['sales', 'revenue', 'temperature', 'humidity', 'rainfall', 'count', 'amount']
                for keyword in value_keywords:
                    if keyword in response_lower:
                        y_label = keyword.title()
                        break
                if not y_label:
                    y_label = "Value"
            
            return {
                'x_values': x_values,
                'y_values': y_values,
                'x_label': x_label,
                'y_label': y_label,
                'unit': unit
            }
        return None
    except Exception as e:
        print(f"ERROR extracting data from response: {str(e)}")
        return None

def extract_visualization_metadata(response_text: str) -> Dict[str, Any]:
    """Extract visualization metadata from Gemini's response using a second LLM call."""
    try:
        prompt = f"""Analyze the following text and extract visualization metadata.
        Please identify:
        1. The main title/heading for the visualization
        2. The x-axis label
        3. The y-axis label
        4. Any units mentioned (%, °C, $, mm, etc.)
        5. Any legend labels/categories mentioned
        
        Format your response as bullet points with clear "label: value" pairs.
        
        Text to analyze:
        {response_text}
        
        Extract only the metadata, don't include any other text or explanations.
        """
        
        print("DEBUG: Sending metadata extraction prompt to Gemini")
        metadata_response = model.generate_content(prompt)
        
        # Initialize metadata dictionary
        metadata = {
            'title': None,
            'x_label': None,
            'y_label': None,
            'unit': None,
            'legend_labels': []
        }
        
        # Parse the response
        for line in metadata_response.text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            # Remove bullet points and clean the line
            line = line.replace('*', '').replace('•', '').strip()
            key, value = [part.strip() for part in line.split(':', 1)]
            
            # Map the extracted metadata
            key_lower = key.lower()
            if 'title' in key_lower or 'heading' in key_lower:
                metadata['title'] = value
            elif 'x-axis' in key_lower or 'x axis' in key_lower:
                metadata['x_label'] = value
            elif 'y-axis' in key_lower or 'y axis' in key_lower:
                metadata['y_label'] = value
            elif 'unit' in key_lower:
                metadata['unit'] = value
            elif 'legend' in key_lower or 'categor' in key_lower:
                metadata['legend_labels'] = [label.strip() for label in value.split(',')]
        
        print(f"DEBUG: Extracted metadata: {metadata}")
        return metadata
    except Exception as e:
        print(f"ERROR extracting visualization metadata: {str(e)}")
        return None

def create_visualization_from_response(response_text: str, query: str) -> Optional[Dict[str, Any]]:
    """Create visualization based on data extracted from Gemini's response."""
    try:
        # Extract data and metadata
        data = extract_data_from_response(response_text)
        if not data:
            print("DEBUG: No data could be extracted from response")
            return None
            
        # Extract metadata using second LLM call
        metadata = extract_visualization_metadata(response_text)
        if metadata:
            # Update data with extracted metadata
            data['title'] = metadata.get('title') or data.get('y_label')
            data['x_label'] = metadata.get('x_label') or data.get('x_label')
            data['y_label'] = metadata.get('y_label') or data.get('y_label')
            data['unit'] = metadata.get('unit') or data.get('unit')
            data['legend_labels'] = metadata.get('legend_labels', [])
            
        print(f"DEBUG: Creating visualization for {data['y_label']} data")
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': data['x_values'],
            'y': data['y_values']
        })
        
        # Detect visualization type
        viz_type = detect_visualization_type(response_text, data, metadata or {})
        print(f"DEBUG: Selected visualization type: {viz_type}")
        
        # Create the plot based on type
        fig = go.Figure()
        
        if viz_type == 'pie':
            fig = go.Figure(data=[go.Pie(
                labels=df['x'],
                values=df['y'],
                hole=0.3,  # Make it a donut chart for better appearance
                textinfo='percent+label',
                hovertemplate="%{label}<br>%{value:,.2f}" + (f" {data['unit']}" if data['unit'] else "") + "<br>%{percent}"
            )])
            
        elif viz_type == 'line':
            # Try to convert x-axis to datetime if possible
            try:
                df['x'] = pd.to_datetime(df['x'])
                df = df.sort_values('x')
            except:
                # If conversion fails, try to handle month names
                month_map = {month: i for i, month in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 
                                                               'july', 'august', 'september', 'october', 'november', 'december'], 1)}
                if all(str(x).lower().strip() in month_map for x in df['x']):
                    df['month_num'] = df['x'].str.lower().map(month_map)
                    df = df.sort_values('month_num')
                    
            fig.add_trace(go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='lines+markers',
                name=data.get('legend_labels')[0] if data.get('legend_labels') else data['y_label'],
                line=dict(width=2),
                marker=dict(size=8)
            ))
            
        else:  # bar chart
            fig.add_trace(go.Bar(
                x=df['x'],
                y=df['y'],
                name=data.get('legend_labels')[0] if data.get('legend_labels') else data['y_label'],
                text=df['y'].apply(lambda x: f"{x:,.2f}" + (f" {data['unit']}" if data['unit'] else "")),
                textposition='auto',
            ))
        
        # Customize layout with extracted metadata
        title = data['title'] or f"{data['y_label']}"
        if data['unit'] and data['unit'] not in title:
            title += f" ({data['unit']})"
            
        fig.update_layout(
            title=title,
            xaxis_title=None if viz_type == 'pie' else data['x_label'],
            yaxis_title=None if viz_type == 'pie' else (f"{data['y_label']}" + (f" ({data['unit']})" if data['unit'] else "")),
            template="plotly_white",
            hovermode='x unified' if viz_type != 'pie' else None,
            showlegend=viz_type == 'pie' or bool(data.get('legend_labels')),
            xaxis=dict(
                tickangle=45 if viz_type == 'bar' else 0,
                type='category' if viz_type == 'bar' else '-'
            )
        )
        
        # Additional layout customizations based on chart type
        if viz_type == 'pie':
            fig.update_layout(
                annotations=[dict(text=data['y_label'], showarrow=False)],
                uniformtext_minsize=12,
                uniformtext_mode='hide'
            )
        elif viz_type == 'bar':
            fig.update_layout(
                bargap=0.2,
                bargroupgap=0.1
            )
        elif viz_type == 'line':
            fig.update_layout(
                hovermode='x unified',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray'
                )
            )
        
        # Convert to HTML
        chart_html = fig.to_html(full_html=False, include_plotlyjs=True)
        print("DEBUG: Successfully created visualization")
        
        return {
            'type': 'visualization',
            'content': chart_html
        }
    except Exception as e:
        print(f"ERROR creating visualization from response: {str(e)}")
        return None

def create_visualization_from_raw_data(raw_data: Dict, query: str) -> Optional[Dict[str, Any]]:
    """Create time series visualization directly from raw data without LLM extraction."""
    try:
        print(f"DEBUG: Creating time series visualization from raw data")
        
        # Create DataFrame from the data
        data = raw_data.get('data', [])
        if not data:
            print("DEBUG: No data found in raw_data")
            return None
            
        columns = raw_data.get('columns', [])
        sheet_name = raw_data.get('sheet_name', 'Unknown')
        
        print(f"DEBUG: Raw data from sheet {sheet_name} with columns: {columns}")
        df = pd.DataFrame(data)
        
        if df.empty:
            print("DEBUG: Empty DataFrame from raw data")
            return None
            
        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: DataFrame columns: {df.columns}")
        
        # Try to find date/time column
        date_col = None
        for col in df.columns:
            # Try to convert to datetime
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # If at least 50% of values are valid dates, consider it a date column
                if df[col].notna().mean() >= 0.5:
                    date_col = col
                    break
            except:
                pass
                
        # If no datetime column found, check for month names or year values
        if not date_col:
            for col in df.columns:
                if df[col].dtype == object:
                    # Check for month names
                    months = ['january', 'february', 'march', 'april', 'may', 'june', 
                             'july', 'august', 'september', 'october', 'november', 'december']
                    values = [str(x).lower().strip() for x in df[col].dropna()]
                    if values and all(any(month in val for month in months) for val in values):
                        date_col = col
                        # Map month names to numbers for sorting
                        month_map = {m: i+1 for i, m in enumerate(months)}
                        # Extract month and create sort key
                        df['_month_num'] = df[col].apply(
                            lambda x: next((month_map[m] for m in months if m in str(x).lower()), None)
                        )
                        # Sort by this key
                        df = df.sort_values('_month_num')
                        break
                        
                # Check for year values
                elif df[col].dtype in (int, float):
                    values = df[col].dropna()
                    # If values look like years (between 1900 and current year + 10)
                    current_year = datetime.now().year
                    if values.min() >= 1900 and values.max() <= current_year + 10:
                        date_col = col
                        # Sort by year
                        df = df.sort_values(col)
                        break
        
        if not date_col:
            print("DEBUG: No date/time column found in data")
            # Try using index as x-axis if no date column
            df = df.sort_index()
            
        # Find numeric columns for y-axis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out any generated helper columns
        numeric_cols = [col for col in numeric_cols if not col.startswith('_')]
        
        if not numeric_cols:
            print("DEBUG: No numeric columns found for y-axis")
            return None
            
        # Try to find the most relevant numeric column based on the query
        query_lower = query.lower()
        value_keywords = ['value', 'amount', 'revenue', 'sales', 'units', 'price', 
                          'quantity', 'rate', 'percentage', 'volume', 'count']
        
        y_col = None
        for keyword in value_keywords:
            if keyword in query_lower:
                matching_cols = [col for col in numeric_cols if keyword.lower() in str(col).lower()]
                if matching_cols:
                    y_col = matching_cols[0]
                    break
                    
        # If no specific match, use the first numeric column
        if not y_col and numeric_cols:
            y_col = numeric_cols[0]
            
        # Create a line chart
        fig = go.Figure()
        
        # Add trace
        if date_col:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(width=3),
                marker=dict(size=8)
            ))
            x_title = date_col
        else:
            # Use index if no date column
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(width=3),
                marker=dict(size=8)
            ))
            x_title = "Index"
            
        # Create title from query
        if 'trend' in query_lower:
            title = f"Trend of {y_col}"
        elif 'over time' in query_lower:
            title = f"{y_col} Over Time"
        else:
            title = f"{y_col} Time Series"
            
        # Add annotations for min and max points
        y_min = df[y_col].min()
        y_max = df[y_col].max()
        y_min_idx = df[y_col].idxmin()
        y_max_idx = df[y_col].idxmax()
        
        if date_col:
            x_min = df.loc[y_min_idx, date_col]
            x_max = df.loc[y_max_idx, date_col]
        else:
            x_min = y_min_idx
            x_max = y_max_idx
            
        # Add annotations for minimum and maximum
        fig.add_annotation(
            x=x_min,
            y=y_min,
            text=f"Min: {y_min:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        fig.add_annotation(
            x=x_max,
            y=y_max,
            text=f"Max: {y_max:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_col,
            hovermode='x unified',
            template='plotly_white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray'
            ),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Convert to HTML
        chart_html = fig.to_html(full_html=False, include_plotlyjs=True)
        print("DEBUG: Successfully created time series visualization from raw data")
        
        return {
            'type': 'visualization',
            'content': chart_html
        }
        
    except Exception as e:
        print(f"DEBUG: Error creating time series visualization from raw data: {str(e)}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return None

def generate_response(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate response using Gemini with context and create visualizations if applicable."""
    print("\nDEBUG: Starting response generation")
    
    # Initialize response
    response_data = {
        'text': '',
        'visualization': None
    }
    
    # Determine if this is a time series trend question
    is_trend_question = any(keyword in query.lower() for keyword in 
                           ['trend', 'over time', 'change', 'timeline', 'historical', 
                            'pattern', 'evolution', 'progression'])
    
    # Process chunks and extract raw data for Excel files
    processed_chunks = []
    for chunk in context_chunks:
        if chunk['metadata'].get('file_type') == 'xlsx':
            # Parse the JSON string content
            content_data = json.loads(chunk['content'])
            print(f"DEBUG: Processing Excel chunk from sheet: {content_data.get('raw_data', {}).get('sheet_name')}")
            processed_chunks.append({
                'content': content_data['description'],
                'metadata': chunk['metadata']
            })
        else:
            processed_chunks.append(chunk)
    
    # Generate text response
    context = "\n\n".join([
        f"Source: {chunk['metadata'].get('file_type', 'unknown')} - "
        f"{chunk['metadata'].get('page', chunk['metadata'].get('paragraph', 'N/A'))}\n"
        f"Content: {chunk['content']}"
        for chunk in processed_chunks
    ])
    
    # Modify prompt based on question type
    if is_trend_question:
        prompt = f"""Answer the user's question using the provided context.

This appears to be a question about trends or patterns over time. Simply respond with:
"Here's a visualization showing {query}"

A chart will be generated automatically from the data - you don't need to describe all the data points.
You can briefly mention any key statistics like minimum, maximum, or average values if they're in the data.

Context:
{context}

Question: {query}

Answer:"""
    else:
        prompt = f"""Answer the user's question using the provided context. If the data shows specific values or distributions, include them in your response.
        When the user asks for a visualization, format the relevant data as bullet points with clear "label: value" pairs.
        For example:
        * Category1: 42.5
        * Category2: 78.3
        
        Don't say "I can't create a visualization" or "I can't see the data" or anything like that.
        Always include units where applicable (%, °C, mm, $, etc.).

    Context:
    {context}

    Question: {query}

    Instructions:
    1. Provide a clear and direct answer based on the actual data
    2. Include specific numbers and values from the data
    3. If visualization is requested, provide the data in a clear bullet-point format with "label: value" pairs
    4. Always include appropriate units with the values

    Answer:"""

    print("\nDEBUG: Sending prompt to Gemini: \n", prompt)
    response = model.generate_content(prompt)
    response_data['text'] = response.text
    print("DEBUG: Got response from Gemini")
    
    # If this is a trend question, use raw data directly for visualization
    if is_trend_question:
        print("DEBUG: Time series trend question detected, using raw data for visualization")
        # Find Excel chunk with raw data
        for chunk in context_chunks:
            if chunk['metadata'].get('file_type') == 'xlsx':
                try:
                    content_data = json.loads(chunk['content'])
                    if 'raw_data' in content_data:
                        print("DEBUG: Found raw data in Excel chunk")
                        viz = create_visualization_from_raw_data(content_data['raw_data'], query)
                        if viz:
                            response_data['visualization'] = viz
                            print("DEBUG: Created time series visualization from raw data")
                            break
                except Exception as e:
                    print(f"DEBUG: Error processing Excel chunk: {str(e)}")
                    continue
    # Otherwise use the original visualization logic
    elif any(keyword in query.lower() for keyword in ['plot', 'graph', 'chart', 'visualize', 'visualization', 'show', 'display']):
        print("DEBUG: Visualization requested, parsing Gemini response")
        viz = create_visualization_from_response(response.text, query)
        if viz:
            response_data['visualization'] = viz
    
    print(f"\nDEBUG: Final response data keys: {response_data.keys()}")
    print(f"DEBUG: Has visualization: {response_data['visualization'] is not None}")
    
    return response_data 