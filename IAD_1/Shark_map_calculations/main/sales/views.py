from django.views.generic import ListView, DetailView
from .models import Sale

class SaleListView(ListView):
    model = Sale
    template_name = 'sales/main.html'
    context_object_name = 'obj'

class SaleDetailView(DetailView):
    model = Sale
    template_name = 'sales/detail.html'