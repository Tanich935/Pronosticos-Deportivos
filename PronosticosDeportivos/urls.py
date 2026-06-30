from django.contrib import admin
from django.urls import path, include # <-- Asegúrate de importar 'include'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('Pronosticador.urls')), # <-- Esto conecta la web!
]