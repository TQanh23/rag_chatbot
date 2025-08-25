from django.urls import path
from .views import AskView

urlpatterns = [
    path('ask/', AskView.as_view()),   # http://localhost:8000/api/ask/
]
