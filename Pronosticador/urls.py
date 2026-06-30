from django.urls import path
from . import views

urlpatterns = [
    # Esta es la ruta raíz de nuestra aplicación
    path('', views.index, name='index'),
]