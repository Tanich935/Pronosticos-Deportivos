from django.shortcuts import render
from .services import pronosticar, EQUIPOS_MUNDIAL_2026

def index(request):
    # Ordenamos los equipos alfabéticamente para los menús desplegables
    equipos = sorted(list(EQUIPOS_MUNDIAL_2026))
    context = {'equipos': equipos}

    if request.method == 'POST':
        local = request.POST.get('local')
        visitante = request.POST.get('visitante')

        if local and visitante:
            if local == visitante:
                context['error'] = "¡Un equipo no puede jugar contra sí mismo!"
            else:
                try:
                    # Pasamos los nombres tal cual vienen del formulario.
                    # El servicio (services.py) ya se encarga de limpiarlos y buscarlos.
                    resultados = pronosticar(local, visitante)
                    
                    # Pasamos la imagen en base64 al HTML
                    context['imagen_b64'] = resultados['imagen_b64']
                    context['local_seleccionado'] = local
                    context['visitante_seleccionado'] = visitante
                except Exception as e:
                    context['error'] = f"Error al procesar la predicción: {str(e)}"

    return render(request, 'Pronosticador/index.html', context)