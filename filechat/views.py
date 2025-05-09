from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.auth.decorators import login_required
from .models import Document, DocumentChunk, ChatSession, ChatMessage
from .utils import (
    detect_file_type, process_pdf, process_csv, process_txt,
    process_docx, process_image, process_excel, store_chunks, query_chunks,
    generate_response
)
import json
import os
from django.conf import settings

@login_required
@ensure_csrf_cookie
def home(request):
    """Render the home page with file upload and chat interface."""
    return render(request, 'filechat/home.html')

@login_required
def upload_file(request):
    """Handle file upload and processing."""
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        reset_chroma = request.POST.get('reset_chroma') == 'true'
        
        try:
            if reset_chroma:
                reset_chroma_collection()
            
            # Save the file
            document = Document.objects.create(
                file=file,
                original_filename=file.name,
                uploaded_by=request.user
            )
            
            # Create media directory if it doesn't exist
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            
            # Detect file type
            file_path = os.path.join(settings.MEDIA_ROOT, document.file.name)
            file_type = detect_file_type(file_path)
            document.file_type = file_type
            document.save()
            
            # Process file based on type
            try:
                if file_type == 'pdf':
                    chunks = process_pdf(file_path)
                elif file_type == 'csv':
                    chunks = process_csv(file_path)
                elif file_type == 'txt':
                    chunks = process_txt(file_path)
                elif file_type == 'docx':
                    chunks = process_docx(file_path)
                elif file_type == 'xlsx':
                    chunks = process_excel(file_path)
                elif file_type == 'image':
                    chunks = process_image(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                # Store chunks in database and vector store
                for i, chunk_data in enumerate(chunks):
                    DocumentChunk.objects.create(
                        document=document,
                        content=chunk_data['content'],
                        page_number=chunk_data.get('page_number'),
                        chunk_index=i,
                        metadata=chunk_data['metadata']
                    )
                
                store_chunks(chunks, str(document.id))
                document.is_processed = True
                document.save()
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'File processed successfully',
                    'document_id': str(document.id)
                })
                
            except Exception as e:
                document.delete()  # Clean up the document if processing fails
                return JsonResponse({
                    'status': 'error',
                    'message': f"Error processing file: {str(e)}"
                }, status=400)
                
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f"Error uploading file: {str(e)}"
            }, status=400)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    }, status=400)

@login_required
def create_chat_session(request):
    """Create a new chat session."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            document_ids = data.get('document_ids', [])
            
            session = ChatSession.objects.create(
                user=request.user,
                title=data.get('title', 'New Chat')
            )
            
            if document_ids:
                documents = Document.objects.filter(id__in=document_ids)
                session.documents.add(*documents)
            
            return JsonResponse({
                'status': 'success',
                'session_id': str(session.id)
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f"Error creating chat session: {str(e)}"
            }, status=400)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    }, status=400)

@login_required
def add_document_to_session(request, session_id, document_id):
    """Add a document to an existing chat session."""
    try:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        document = get_object_or_404(Document, id=document_id, uploaded_by=request.user)
        
        session.documents.add(document)
        
        return JsonResponse({
            'status': 'success',
            'message': 'Document added to session successfully'
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

@login_required
def send_message(request, session_id):
    """Handle chat messages and generate responses."""
    if request.method == 'POST':
        try:
            print("DEBUG: Starting message processing")
            session = ChatSession.objects.get(id=session_id, user=request.user)
            data = json.loads(request.body)
            message = data.get('message')
            
            if not message:
                raise ValueError("Message is required")
            
            print(f"DEBUG: Processing message: {message}")
            
            # Save user message
            ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message
            )
            
            # Get relevant chunks from documents in the session
            document_ids = [str(doc.id) for doc in session.documents.all()]
            print(f"DEBUG: Document IDs in session: {document_ids}")
            
            relevant_chunks = query_chunks(message)
            print(f"DEBUG: Found {len(relevant_chunks)} relevant chunks")
            
            # Filter chunks to only include documents in the session
            relevant_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['metadata']['document_id'] in document_ids
            ]
            print(f"DEBUG: After filtering, {len(relevant_chunks)} chunks remain")
            
            if not relevant_chunks:
                return JsonResponse({
                    'status': 'error',
                    'message': 'No relevant information found in the uploaded documents.'
                })
            
            # Generate response
            print("DEBUG: Calling generate_response")
            response_data = generate_response(message, relevant_chunks)
            print("DEBUG: Response generated successfully")
            print(f"DEBUG: Response data keys: {response_data.keys()}")
            
            # Save assistant response
            ChatMessage.objects.create(
                session=session,
                message_type='assistant',
                content=response_data['text'],  # Store only the text part
                sources=[chunk['metadata'] for chunk in relevant_chunks]
            )
            
            return JsonResponse({
                'status': 'success',
                'response': response_data,  # Return the complete response data
                'sources': [chunk['metadata'] for chunk in relevant_chunks]
            })
            
        except ChatSession.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Chat session not found'
            }, status=404)
        except Exception as e:
            print(f"DEBUG: Error in send_message: {str(e)}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    }, status=400)

@login_required
def get_chat_history(request, session_id):
    """Get chat history for a session."""
    try:
        session = ChatSession.objects.get(id=session_id, user=request.user)
        messages = ChatMessage.objects.filter(session=session)
        
        return JsonResponse({
            'status': 'success',
            'messages': [
                {
                    'type': msg.message_type,
                    'content': msg.content,
                    'created_at': msg.created_at.isoformat(),
                    'sources': msg.sources
                }
                for msg in messages
            ]
        })
        
    except ChatSession.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Chat session not found'
        }, status=404)
