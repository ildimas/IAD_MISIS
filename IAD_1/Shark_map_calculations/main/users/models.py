from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    phone_numder = models.CharField(max_length=12)
    is_pidor = models.BooleanField(default=True)