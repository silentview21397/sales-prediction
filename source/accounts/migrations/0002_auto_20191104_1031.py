# Generated by Django 2.1.1 on 2019-11-04 05:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='HouseDetail',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('owner_name', models.CharField(max_length=100)),
                ('price_range', models.CharField(max_length=100)),
                ('building_floors_square_feet', models.PositiveIntegerField()),
                ('zone_category', models.CharField(choices=[('1', 'Rural'), ('2', 'Urban')], default='Rural', max_length=6)),
                ('distance_from_road', models.PositiveIntegerField()),
                ('total_area_covered', models.PositiveIntegerField()),
                ('population_in_area', models.PositiveIntegerField()),
                ('house_rating', models.PositiveIntegerField()),
                ('material_rating', models.PositiveIntegerField()),
                ('room_type', models.CharField(choices=[('1', '1 BHK'), ('2', '2 BHK'), ('3', '3 BHK')], default='1 BHK', max_length=6)),
                ('neighbourhood_class', models.CharField(choices=[('1', 'Poor'), ('2', 'Medium Class'), ('3', 'Rich'), ('4', 'Very Rich')], default='Poor', max_length=6)),
                ('year_built', models.PositiveIntegerField()),
                ('year_rebuilt', models.PositiveIntegerField()),
                ('roof_material', models.PositiveIntegerField()),
                ('basement_quality', models.PositiveIntegerField()),
                ('heating', models.PositiveIntegerField()),
                ('bathroom_type', models.CharField(choices=[('1', 'Indian'), ('2', 'Western')], default='Indian', max_length=6)),
                ('kitchen_quality', models.PositiveIntegerField()),
                ('house_image', models.FileField(upload_to='house_image')),
                ('room_image', models.FileField(upload_to='room_image')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'house_details',
            },
        ),
        migrations.RemoveField(
            model_name='uploadeddocuments',
            name='user',
        ),
        migrations.DeleteModel(
            name='UploadedDocuments',
        ),
    ]
