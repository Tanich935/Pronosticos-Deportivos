from django.db import models

class Equipo(models.Model):
    nombre = models.CharField(max_length=100, unique=True)
    
    def __str__(self):
        return self.nombre

class Partido(models.Model):
    fecha = models.DateField()
    equipo_local = models.ForeignKey(Equipo, related_name='partidos_local', on_delete=models.CASCADE)
    equipo_visitante = models.ForeignKey(Equipo, related_name='partidos_visitante', on_delete=models.CASCADE)
    goles_local = models.IntegerField()
    goles_visitante = models.IntegerField()
    torneo = models.CharField(max_length=100)
    neutral = models.BooleanField(default=False, help_text="¿Se jugó en cancha neutral?")

    def __str__(self):
        return f"{self.fecha}: {self.equipo_local} {self.goles_local} - {self.goles_visitante} {self.equipo_visitante}"