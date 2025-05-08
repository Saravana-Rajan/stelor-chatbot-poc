from django.db import models
from django.contrib.auth.models import User
import uuid

class Document(models.Model):
    FILE_TYPES = (
        ('pdf', 'PDF'),
        ('csv', 'CSV'),
        ('txt', 'Text'),
        ('docx', 'Word'),
        ('xlsx', 'Excel'),
        ('image', 'Image'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(upload_to='documents/')
    file_type = models.CharField(max_length=10, choices=FILE_TYPES)
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    is_processed = models.BooleanField(default=False)
    metadata = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"{self.original_filename} ({self.file_type})"

class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    page_number = models.IntegerField(null=True, blank=True)
    chunk_index = models.IntegerField()
    embedding = models.JSONField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['document', 'chunk_index']

class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    documents = models.ManyToManyField(Document, related_name='chat_sessions')
    title = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"Chat Session {self.id} - {self.user.username}"

class ChatMessage(models.Model):
    MESSAGE_TYPES = (
        ('user', 'User Message'),
        ('assistant', 'Assistant Response'),
    )

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    sources = models.JSONField(default=list, blank=True)  # List of document references used in the response

    class Meta:
        ordering = ['created_at']
