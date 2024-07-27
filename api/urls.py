from django.urls import path
from .views import PredictView, GuideView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('', GuideView.as_view(), name='guide'),
]
