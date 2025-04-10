from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser
from .forms import CustomUserCreationForm

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    add_form = CustomUserCreationForm

    fieldsets = (
        *UserAdmin.fieldsets,
        (
            'Phone number',
            {
                'fields': (
                    'phone_numder',
                )
            }
        )
    )

admin.site.register(CustomUser, CustomUserAdmin)
