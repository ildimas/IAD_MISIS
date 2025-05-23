# Generated by Django 4.1.3 on 2022-12-13 22:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='main_db',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('estate_type', models.CharField(max_length=200, verbose_name='Тип недвижимости')),
                ('link', models.TextField(verbose_name='Ссылка')),
                ('address', models.CharField(max_length=200, verbose_name='Адрес')),
                ('flat_floor', models.IntegerField(verbose_name='Этаж квартиры')),
                ('building_floor', models.IntegerField(verbose_name='Этажность здания')),
                ('price', models.IntegerField(verbose_name='Цена квартиры')),
                ('rooms_count', models.IntegerField(verbose_name='Количество комнат')),
                ('kithcen_square', models.IntegerField(verbose_name='Площадь кухни')),
                ('main_square', models.IntegerField(verbose_name='Площадь квартиры')),
                ('balcony', models.BooleanField(default=False, verbose_name='Наличие балкона')),
                ('decor', models.CharField(max_length=150, verbose_name='Тип ремонта')),
                ('subway_distance', models.IntegerField(verbose_name='расстояние до метро')),
                ('apartment_type', models.CharField(max_length=200, verbose_name='Тип дома')),
                ('coordinates_lng', models.DecimalField(decimal_places=20, max_digits=30, verbose_name='Долгота')),
                ('coordinates_lat', models.DecimalField(decimal_places=20, max_digits=30, verbose_name='Широта')),
            ],
        ),
        migrations.CreateModel(
            name='uploaded_files',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('estate_type', models.CharField(max_length=200, verbose_name='Сегмент')),
                ('address', models.CharField(max_length=200, verbose_name='Адрес')),
                ('flat_floor', models.IntegerField(verbose_name='Этаж квартиры')),
                ('building_floor', models.IntegerField(verbose_name='Этажность здания')),
                ('rooms_count', models.IntegerField(verbose_name='Количество комнат')),
                ('kithcen_square', models.IntegerField(verbose_name='Площадь кухни')),
                ('main_square', models.IntegerField(verbose_name='Площадь квартиры')),
                ('balcony', models.BooleanField(default=False, verbose_name='Наличие балкона')),
                ('decor', models.CharField(max_length=150, verbose_name='Тип ремонта')),
                ('subway_distance', models.IntegerField(verbose_name='расстояние до метро')),
                ('apartment_type', models.CharField(max_length=200, verbose_name='Материал стен')),
                ('coordinates_lng', models.DecimalField(decimal_places=20, max_digits=30, verbose_name='Долгота')),
                ('coordinates_lat', models.DecimalField(decimal_places=20, max_digits=30, verbose_name='Широта')),
            ],
        ),
    ]
