from django.db import models

class Doc(models.Model):
    upload = models.ImageField(upload_to='files/')

    def __str__(self):
        return str(self.pk)