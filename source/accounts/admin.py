from django.contrib import admin
from .models import HouseDetail

admin.site.register(HouseDetail)
# admin.site.register(Author)
# admin.site.register(Book)
from django.contrib.admin import AdminSite
from django.utils.translation import ugettext_lazy



admin.site.site_header = 'House Management System Admin'
admin.site.site_title = 'House Management System Admin'
admin.site.index_title = 'Administration'