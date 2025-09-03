from django.urls import path, include
from .views import AskView, FileUploadView, InitQdrantView, DeleteQdrantCollectionView

urlpatterns = []

urlpatterns = [
    #path('api/', include('api.urls')),  # Include the API app's URLs
    path('ask/', AskView.as_view()),   # http://localhost:8000/api/ask/
    path('upload/', FileUploadView.as_view(), name='file-upload'),  # New upload endpoint
    #path('upload/pdf/', PDFUploadView.as_view(), name='pdf-upload'),
    path('init-qdrant/', InitQdrantView.as_view(), name='init-qdrant'),
    path('delete-qdrant/', DeleteQdrantCollectionView.as_view())
]
