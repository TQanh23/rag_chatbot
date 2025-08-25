from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response

class AskView(APIView):
    def post(self, request):
        question = request.data.get("question")
        return Response({"answer": f"Echo: {question}"})
