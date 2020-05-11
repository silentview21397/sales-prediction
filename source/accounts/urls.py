from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import (
    LogInView, ResendActivationCodeView, RemindUsernameView, SignUpView, ActivateView, LogOutView,
    ChangeEmailView, ChangeEmailActivateView, ChangeProfileView, ChangePasswordView,
    RestorePasswordView, RestorePasswordDoneView, RestorePasswordConfirmView
)
from . import views
app_name = 'accounts'
from django.views.generic import TemplateView
urlpatterns = [
    path('log-in/', LogInView.as_view(), name='log_in'),
    path('log-out/', LogOutView.as_view(), name='log_out'),

    path('resend/activation-code/', ResendActivationCodeView.as_view(), name='resend_activation_code'),

    path('sign-up/', SignUpView.as_view(), name='sign_up'),
    path('activate/<code>/', ActivateView.as_view(), name='activate'),

    path('restore/password/', RestorePasswordView.as_view(), name='restore_password'),
    path('restore/password/done/', RestorePasswordDoneView.as_view(), name='restore_password_done'),
    path('restore/<uidb64>/<token>/', RestorePasswordConfirmView.as_view(), name='restore_password_confirm'),

    path('remind/username/', RemindUsernameView.as_view(), name='remind_username'),

    path('change/profile/', ChangeProfileView.as_view(), name='change_profile'),
    path('change/password/', ChangePasswordView.as_view(), name='change_password'),
    path('change/email/', ChangeEmailView.as_view(), name='change_email'),
    path('change/email/<code>/', ChangeEmailActivateView.as_view(), name='change_email_activation'),
    path('searchHouse/', views.searchHouse, name='searchHouse'),
    path('listHouses/', views.AllHousesDetails.as_view(), name='listHouses'),
    path('gridSearch/', views.searchHouseGrid, name='gridSearch'),
    path('predictHouse/', views.predictHouse, name='predictHouse'),
    path('accuracyTest/', views.accuracyTest, name='accuracyTest'),
    path('profile/', TemplateView.as_view(template_name='profile.html'), name='profile'),
    path('predict/', TemplateView.as_view(template_name='prediction.html'), name='predict'),
    path('accuracy/', TemplateView.as_view(template_name='accuracy.html'), name='accuracy'),
    path('grid/', TemplateView.as_view(template_name='house_grid.html'), name='grid'),



]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
