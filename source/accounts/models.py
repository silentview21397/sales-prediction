from django.db import models
from django.contrib.auth.models import User


class Activation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    code = models.CharField(max_length=20, unique=True)
    email = models.EmailField(blank=True)

from django.core.validators import RegexValidator
class HouseDetail(models.Model):
    # output

    zone = (
        ('1', 'Rural'),
        ('2', 'Urban'),

    )
    room = (
        ('1', '1 BHK'),
        ('2', '2 BHK'),
        ('3', '3 BHK'),

    )
    bathroom = (
        ('1', 'Indian'),
        ('2', 'Western'),


    )

    neighbourhood = (
        ('1', 'Poor'),
        ('2', 'Medium Class'),
        ('3', 'Rich'),
        ('4', 'Very Rich'),

    )
    owner_name = models.CharField(max_length=100)
    price_range = models.CharField(max_length=100)
    building_floors_square_feet = models.PositiveIntegerField()
    zone_category=models.CharField(max_length=6, choices=zone, default='Rural')
    distance_from_road=models.PositiveIntegerField();
    total_area_covered=models.PositiveIntegerField();
    room_type=models.CharField(max_length=6,choices=room,default="1 BHK");
    population_in_area = models.PositiveIntegerField();
    # 1-5 for best
    house_rating = models.PositiveIntegerField();
    # 1-5 for best
    material_rating = models.PositiveIntegerField();
    room_type=models.CharField(max_length=6,choices=room,default="1 BHK");
    neighbourhood_class=models.CharField(max_length=6,choices=neighbourhood,default="Poor");
    year_built=models.PositiveIntegerField()
    year_rebuilt=models.PositiveIntegerField()
    # 1-5 for best
    roof_material=models.PositiveIntegerField()
    # 1-5
    basement_quality=models.PositiveIntegerField()
    # 5-1
    heating=models.PositiveIntegerField()
    bathroom_type = models.CharField(max_length=6, choices=bathroom, default="Indian");
    # 1-5
    kitchen_quality = models.PositiveIntegerField();
    house_image = models.FileField(upload_to='house_image')
    room_image = models.FileField(upload_to='room_image')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.owner_name
    class Meta:
        db_table = "house_details"

