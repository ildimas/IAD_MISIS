from django import forms 

class ExcellForm(forms.Form):
    adress = forms.CharField(max_length=200)
    rooms_count = forms.IntegerField()
    estate_type = forms.CharField(max_length=200)
    buildeing_floor_count = forms.IntegerField()
    apartment_materials = forms.CharField(max_length=200)
    apartment_floor = forms.IntegerField()
    apartment_square = forms.IntegerField()
    kitchen_square = forms.IntegerField()
    balcony = forms.BooleanField()
    subway_distance = forms.IntegerField()
    decor = forms.CharField(max_length=200)