from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import auth_view, verify_view, index, upload_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('login/', auth_view, name='login'),
    path('verify/', verify_view, name='verify'),
    path('', index, name='home'),
    path('upload/', upload_view, name='upload'),
    path('', include('sales.urls', namespace='sales')),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)