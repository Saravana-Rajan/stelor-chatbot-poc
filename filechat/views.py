from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import Document, DocumentChunk, ChatSession, ChatMessage
from .utils import (
    detect_file_type, process_pdf, process_csv, process_txt,
    process_docx, process_image, process_excel, store_chunks, query_chunks,
    generate_response
)
import json
import os
from django.conf import settings

def get_default_user():
    """Get or create a default user for development purposes."""
    user, created = User.objects.get_or_create(
        username='dev_user',
        defaults={
            'email': 'dev@example.com',
            'first_name': 'Dev',
            'last_name': 'User'
        }
    )
    return user

@ensure_csrf_cookie
def home(request):
    """Render the home page with file upload and chat interface."""
    return render(request, 'filechat/home.html')

def upload_file(request):
    """Handle file upload and processing."""
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        reset_chroma = request.POST.get('reset_chroma') == 'true'
        
        try:
            if reset_chroma:
                reset_chroma_collection()
            
            # Save the file
            user = get_default_user()
            document = Document.objects.create(
                file=file,
                original_filename=file.name,
                uploaded_by=user
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
                
                # Create a new chat session specifically for this document
                session = ChatSession.objects.create(
                    user=user,
                    title=f"Chat with {document.original_filename}"
                )
                session.documents.add(document)
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'File processed successfully',
                    'document_id': str(document.id),
                    'session_id': str(session.id)
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

def create_chat_session(request):
    """Create a new chat session."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            document_ids = data.get('document_ids', [])
            
            user = get_default_user()
            session = ChatSession.objects.create(
                user=user,
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

def add_document_to_session(request, session_id, document_id):
    """Add a document to an existing chat session."""
    try:
        user = get_default_user()
        session = get_object_or_404(ChatSession, id=session_id, user=user)
        document = get_object_or_404(Document, id=document_id, uploaded_by=user)
        
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

def send_message(request, session_id):
    """Handle chat messages and generate responses."""
    if request.method == 'POST':
        try:
            print("DEBUG: Starting message processing")
            user = get_default_user()
            session = ChatSession.objects.get(id=session_id, user=user)
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
            
            # Use direct filtering in the query instead of post-filtering
            relevant_chunks = query_chunks(message, document_ids=document_ids)
            print(f"DEBUG: Found {len(relevant_chunks)} relevant chunks")
            
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

def get_chat_history(request, session_id):
    """Get chat history for a session."""
    try:
        user = get_default_user()
        session = ChatSession.objects.get(id=session_id, user=user)
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

def list_chat_sessions(request):
    """List all chat sessions for the current user."""
    try:
        user = get_default_user()
        sessions = ChatSession.objects.filter(user=user).order_by('-created_at')
        
        sessions_data = []
        for session in sessions:
            # Get the latest message for preview
            latest_message = session.messages.order_by('-created_at').first()
            message_count = session.messages.count()
            
            sessions_data.append({
                'id': str(session.id),
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'message_count': message_count,
                'latest_message': latest_message.content[:100] + '...' if latest_message and len(latest_message.content) > 100 else latest_message.content if latest_message else None,
                'documents': [
                    {
                        'id': str(doc.id),
                        'filename': doc.original_filename,
                        'file_type': doc.file_type
                    }
                    for doc in session.documents.all()
                ]
            })
        
        return JsonResponse({
            'status': 'success',
            'sessions': sessions_data
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=400)

def delete_chat_session(request, session_id):
    """Delete a chat session."""
    if request.method == 'DELETE':
        try:
            user = get_default_user()
            session = ChatSession.objects.get(id=session_id, user=user)
            session.delete()
            
            return JsonResponse({
                'status': 'success',
                'message': 'Chat session deleted successfully'
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
        'message': 'Invalid request method'
    }, status=405)
