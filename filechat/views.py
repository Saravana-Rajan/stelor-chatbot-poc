from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from .models import Document, DocumentChunk, ChatSession, ChatMessage
from .utils import (
    detect_file_type, process_pdf, process_csv, process_txt,
    process_docx, process_image, store_chunks, query_chunks,
    generate_response
)
import json
import os
from django.conf import settings

@login_required
def home(request):
    """Render the home page with file upload and chat interface."""
    return render(request, 'filechat/home.html')

@login_required
@csrf_exempt
def upload_file(request):
    """Handle file upload and processing."""
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        
        # Save the file
        document = Document.objects.create(
            file=file,
            original_filename=file.name,
            uploaded_by=request.user
        )
        
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
            document.delete()
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    }, status=400)

@login_required
@csrf_exempt
def create_chat_session(request):
    """Create a new chat session."""
    if request.method == 'POST':
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
    
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request'
    }, status=400)

@login_required
@csrf_exempt
def send_message(request, session_id):
    """Handle chat messages and generate responses."""
    if request.method == 'POST':
        try:
            session = ChatSession.objects.get(id=session_id, user=request.user)
            data = json.loads(request.body)
            message = data.get('message')
            
            if not message:
                raise ValueError("Message is required")
            
            # Save user message
            ChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message
            )
            
            # Get relevant chunks from documents in the session
            document_ids = [str(doc.id) for doc in session.documents.all()]
            relevant_chunks = query_chunks(message)
            
            # Filter chunks to only include documents in the session
            relevant_chunks = [
                chunk for chunk in relevant_chunks
                if chunk['metadata']['document_id'] in document_ids
            ]
            
            # Generate response
            response = generate_response(message, relevant_chunks)
            
            # Save assistant response
            ChatMessage.objects.create(
                session=session,
                message_type='assistant',
                content=response,
                sources=[chunk['metadata'] for chunk in relevant_chunks]
            )
            
            return JsonResponse({
                'status': 'success',
                'response': response,
                'sources': [chunk['metadata'] for chunk in relevant_chunks]
            })
            
        except ChatSession.DoesNotExist:
            return JsonResponse({
                'status': 'error',
                'message': 'Chat session not found'
            }, status=404)
        except Exception as e:
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
