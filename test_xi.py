import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'PronosticosDeportivos.settings')
django.setup()

from Pronosticador.services import calcularFuerzasEquipos

for xi_val in [0.001, 0.002, 0.003, 0.004]:
    f, p = calcularFuerzasEquipos(xi=xi_val)
    ranking = f.sort_values('Ataque', ascending=False).index.tolist()
    top5 = ranking[:5]
    arg_pos = ranking.index('Argentina') + 1
    bra_pos = ranking.index('Brazil') + 1
    esp_pos = ranking.index('Spain') + 1
    fra_pos = ranking.index('France') + 1
    print(f"\nxi={xi_val}")
    print(f"  Top 5: {top5}")
    print(f"  Francia #{fra_pos} | España #{esp_pos} | Argentina #{arg_pos} | Brazil #{bra_pos}")

