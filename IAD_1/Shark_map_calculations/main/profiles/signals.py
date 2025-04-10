from django.db.models.signals import post_save
from django.dispatch import receiver
from users.models import CustomUser
from .models import Profile

@receiver(post_save, sender=CustomUser)
def post_save_create_profile(sender, instance, created, **kwargs):
    print(sender, instance, created)
    if created:
        Profile.objects.create(user=instance)
