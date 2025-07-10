from django.urls import path
from . import views

app_name = 'filechat'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload_file'),
    path('chat/create/', views.create_chat_session, name='create_chat_session'),
    path('chat/<uuid:session_id>/message/', views.send_message, name='send_message'),
    path('chat/<uuid:session_id>/history/', views.get_chat_history, name='get_chat_history'),
    path('chat/session/<uuid:session_id>/add_document/<uuid:document_id>/', views.add_document_to_session, name='add_document_to_session'),
    path('chat/sessions/', views.list_chat_sessions, name='list_chat_sessions'),
    path('chat/session/<uuid:session_id>/delete/', views.delete_chat_session, name='delete_chat_session'),
] 