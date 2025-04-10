from django.urls import path
from .views import SaleListView, SaleDetailView

app_name = 'sales'

urlpatterns = [
    path('sales/', SaleListView.as_view(), name='list'),
    path('sales/<pk>/', SaleDetailView.as_view(), name='detail'),
]