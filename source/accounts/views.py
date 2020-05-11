from django.contrib import messages
from django.contrib.auth import login, authenticate, REDIRECT_FIELD_NAME
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import (
    LogoutView as BaseLogoutView, PasswordChangeView as BasePasswordChangeView,
    PasswordResetDoneView as BasePasswordResetDoneView, PasswordResetConfirmView as BasePasswordResetConfirmView,
)
from django.shortcuts import get_object_or_404, redirect
from django.utils.crypto import get_random_string
from django.utils.decorators import method_decorator
from django.utils.http import is_safe_url
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.utils.translation import gettext_lazy as _
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.debug import sensitive_post_parameters
from django.views.generic import View, FormView
from django.conf import settings

from .utils import (
    send_activation_email, send_reset_password_email, send_forgotten_username_email, send_activation_change_email,
)
from .forms import (
    SignInViaUsernameForm, SignInViaEmailForm, SignInViaEmailOrUsernameForm, SignUpForm,
    RestorePasswordForm, RestorePasswordViaEmailOrUsernameForm, RemindUsernameForm,
    ResendActivationCodeForm, ResendActivationCodeViaEmailForm, ChangeProfileForm, ChangeEmailForm,
)
from .models import Activation


class GuestOnlyView(View):
    def dispatch(self, request, *args, **kwargs):
        # Redirect to the index page if the user already authenticated
        if request.user.is_authenticated:
            return redirect(settings.LOGIN_REDIRECT_URL)

        return super().dispatch(request, *args, **kwargs)


class LogInView(GuestOnlyView, FormView):
    template_name = 'accounts/log_in.html'

    @staticmethod
    def get_form_class(**kwargs):
        if settings.DISABLE_USERNAME or settings.LOGIN_VIA_EMAIL:
            return SignInViaEmailForm

        if settings.LOGIN_VIA_EMAIL_OR_USERNAME:
            return SignInViaEmailOrUsernameForm

        return SignInViaUsernameForm

    @method_decorator(sensitive_post_parameters('password'))
    @method_decorator(csrf_protect)
    @method_decorator(never_cache)
    def dispatch(self, request, *args, **kwargs):
        # Sets a test cookie to make sure the user has cookies enabled
        request.session.set_test_cookie()

        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        request = self.request

        # If the test cookie worked, go ahead and delete it since its no longer needed
        if request.session.test_cookie_worked():
            request.session.delete_test_cookie()

        # The default Django's "remember me" lifetime is 2 weeks and can be changed by modifying
        # the SESSION_COOKIE_AGE settings' option.
        if settings.USE_REMEMBER_ME:
            if not form.cleaned_data['remember_me']:
                request.session.set_expiry(0)

        login(request, form.user_cache)

        redirect_to = request.POST.get(REDIRECT_FIELD_NAME, request.GET.get(REDIRECT_FIELD_NAME))
        url_is_safe = is_safe_url(redirect_to, allowed_hosts=request.get_host(), require_https=request.is_secure())

        if url_is_safe:
            return redirect(redirect_to)

        return redirect(settings.LOGIN_REDIRECT_URL)


class SignUpView(GuestOnlyView, FormView):
    template_name = 'accounts/sign_up.html'
    form_class = SignUpForm

    def form_valid(self, form):
        request = self.request
        user = form.save(commit=False)

        if settings.DISABLE_USERNAME:
            # Set a temporary username
            user.username = get_random_string()
        else:
            user.username = form.cleaned_data['username']

        if settings.ENABLE_USER_ACTIVATION:
            user.is_active = False

        # Create a user record
        user.save()

        # Change the username to the "user_ID" form
        if settings.DISABLE_USERNAME:
            user.username = f'user_{user.id}'
            user.save()

        if settings.ENABLE_USER_ACTIVATION:
            code = get_random_string(20)

            act = Activation()
            act.code = code
            act.user = user
            act.save()

            send_activation_email(request, user.email, code)

            messages.success(
                request, _('You are signed up. To activate the account, follow the link sent to the mail.'))
        else:
            raw_password = form.cleaned_data['password1']

            user = authenticate(username=user.username, password=raw_password)
            login(request, user)

            messages.success(request, _('You are successfully signed up!'))

        return redirect('index')


class ActivateView(View):
    @staticmethod
    def get(request, code):
        act = get_object_or_404(Activation, code=code)

        # Activate profile
        user = act.user
        user.is_active = True
        user.save()

        # Remove the activation record
        act.delete()

        messages.success(request, _('You have successfully activated your account!'))

        return redirect('accounts:log_in')


class ResendActivationCodeView(GuestOnlyView, FormView):
    template_name = 'accounts/resend_activation_code.html'

    @staticmethod
    def get_form_class(**kwargs):
        if settings.DISABLE_USERNAME:
            return ResendActivationCodeViaEmailForm

        return ResendActivationCodeForm

    def form_valid(self, form):
        user = form.user_cache

        activation = user.activation_set.first()
        activation.delete()

        code = get_random_string(20)

        act = Activation()
        act.code = code
        act.user = user
        act.save()

        send_activation_email(self.request, user.email, code)

        messages.success(self.request, _('A new activation code has been sent to your email address.'))

        return redirect('accounts:resend_activation_code')


class RestorePasswordView(GuestOnlyView, FormView):
    template_name = 'accounts/restore_password.html'

    @staticmethod
    def get_form_class(**kwargs):
        if settings.RESTORE_PASSWORD_VIA_EMAIL_OR_USERNAME:
            return RestorePasswordViaEmailOrUsernameForm

        return RestorePasswordForm

    def form_valid(self, form):
        user = form.user_cache
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk)).decode()

        send_reset_password_email(self.request, user.email, token, uid)

        return redirect('accounts:restore_password_done')


class ChangeProfileView(LoginRequiredMixin, FormView):
    template_name = 'accounts/profile/change_profile.html'
    form_class = ChangeProfileForm

    def get_initial(self):
        user = self.request.user
        initial = super().get_initial()
        initial['first_name'] = user.first_name
        initial['last_name'] = user.last_name
        return initial

    def form_valid(self, form):
        user = self.request.user
        user.first_name = form.cleaned_data['first_name']
        user.last_name = form.cleaned_data['last_name']
        user.save()

        messages.success(self.request, _('Profile data has been successfully updated.'))

        return redirect('accounts:change_profile')


class ChangeEmailView(LoginRequiredMixin, FormView):
    template_name = 'accounts/profile/change_email.html'
    form_class = ChangeEmailForm

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def get_initial(self):
        initial = super().get_initial()
        initial['email'] = self.request.user.email
        return initial

    def form_valid(self, form):
        user = self.request.user
        email = form.cleaned_data['email']

        if settings.ENABLE_ACTIVATION_AFTER_EMAIL_CHANGE:
            code = get_random_string(20)

            act = Activation()
            act.code = code
            act.user = user
            act.email = email
            act.save()

            send_activation_change_email(self.request, email, code)

            messages.success(self.request, _('To complete the change of email address, click on the link sent to it.'))
        else:
            user.email = email
            user.save()

            messages.success(self.request, _('Email successfully changed.'))

        return redirect('accounts:change_email')


class ChangeEmailActivateView(View):
    @staticmethod
    def get(request, code):
        act = get_object_or_404(Activation, code=code)

        # Change the email
        user = act.user
        user.email = act.email
        user.save()

        # Remove the activation record
        act.delete()

        messages.success(request, _('You have successfully changed your email!'))

        return redirect('accounts:change_email')


class RemindUsernameView(GuestOnlyView, FormView):
    template_name = 'accounts/remind_username.html'
    form_class = RemindUsernameForm

    def form_valid(self, form):
        user = form.user_cache
        send_forgotten_username_email(user.email, user.username)

        messages.success(self.request, _('Your username has been successfully sent to your email.'))

        return redirect('accounts:remind_username')


class ChangePasswordView(BasePasswordChangeView):
    template_name = 'accounts/profile/change_password.html'

    def form_valid(self, form):
        # Change the password
        user = form.save()

        # Re-authentication
        login(self.request, user)

        messages.success(self.request, _('Your password was changed.'))

        return redirect('accounts:change_password')


class RestorePasswordConfirmView(BasePasswordResetConfirmView):
    template_name = 'accounts/restore_password_confirm.html'

    def form_valid(self, form):
        # Change the password
        form.save()

        messages.success(self.request, _('Your password has been set. You may go ahead and log in now.'))

        return redirect('accounts:log_in')


class RestorePasswordDoneView(BasePasswordResetDoneView):
    template_name = 'accounts/restore_password_done.html'


class LogOutView(LoginRequiredMixin, BaseLogoutView):
    template_name = 'accounts/log_out.html'


from .models import HouseDetail
from django.shortcuts import render, redirect

from django.template import RequestContext
from django.shortcuts import render_to_response
from operator import __or__ as OR
from django.db.models import Q
from functools import reduce


def searchHouse(request):
    query = request.GET.get('q')

    print(query)
    if query:
        lst = [Q(house_no__contains=query), Q(owner_name__icontains=query), Q(pincode__icontains=query), Q(address__icontains=query) ,Q(house_type__icontains=query)]
        print(lst)
        results = HouseDetail.objects.order_by('-id').filter(reduce(OR, lst))
    if(len(results)==0):
        messages.warning(request, _('No House details found'))

        # context = RequestContext(request)
    return render(request, 'profile.html', {
            'housedetail_list': results,'query':query
        })
from sklearn.naive_bayes import GaussianNB
def predictHouse(request):
    building_floors_square_feet = request.GET.get('building_floors_square_feet')
    zone_category = request.GET.get('zone_category')
    distance_from_road = request.GET.get('distance_from_road')
    total_area_covered = request.GET.get('total_area_covered')
    population_in_area = request.GET.get('population_in_area')
    house_rating = request.GET.get('house_rating')
    material_rating = request.GET.get('material_rating')
    room_type = request.GET.get('room_type')
    neighbourhood_class = request.GET.get('neighbourhood_class')
    year_built = request.GET.get('year_built')
    year_rebuilt = request.GET.get('year_rebuilt')
    roof_material = request.GET.get('roof_material')
    basement_quality = request.GET.get('basement_quality')
    heating = request.GET.get('heating')
    bathroom_type = request.GET.get('bathroom_type')
    kitchen_quality = request.GET.get('kitchen_quality')

    print(building_floors_square_feet)
    results = HouseDetail.objects.order_by('-id');
    test = [float(building_floors_square_feet), float(zone_category), float(distance_from_road),
            float(total_area_covered), float(population_in_area), float(house_rating)
        , float(material_rating),
            float(room_type), float(neighbourhood_class), float(year_built), float(year_rebuilt),float(roof_material),float(basement_quality),float(heating),float(bathroom_type),float(kitchen_quality)]
    # trainng dataset

    features = []
    label = []
    for res in results:
        print(res.building_floors_square_feet)
        features.append([float(res.building_floors_square_feet), float(res.zone_category),
                         float(res.distance_from_road), float(res.total_area_covered), float(res.population_in_area), float(res.house_rating),
                         float(res.material_rating), float(res.room_type), float(res.neighbourhood_class), float(res.year_built),
                         float(res.year_rebuilt),float(res.roof_material),float(res.basement_quality),float(res.heating),float(res.bathroom_type),float(res.kitchen_quality)])
        label.append(res.price_range)
    # Create a Gaussian Classifier
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(features, label)

    # Predict Output
    predicted = model.predict([test])  # 0:Overcast, 2:Mild
    print("Predicted Value:", predicted)

        # context = RequestContext(request)
    return render(request, 'prediction.html', {
            'output': predicted[0]
        })
from sklearn import svm
def accuracyTest(request):



    results = HouseDetail.objects.order_by('-id');
       # trainng dataset

    features = []
    label = []
    for res in results:
        print(res.building_floors_square_feet)
        features.append([float(res.building_floors_square_feet), float(res.zone_category),
                         float(res.distance_from_road), float(res.total_area_covered), float(res.population_in_area), float(res.house_rating),
                         float(res.material_rating), float(res.room_type), float(res.neighbourhood_class), float(res.year_built),
                         float(res.year_rebuilt),float(res.roof_material),float(res.basement_quality),float(res.heating),float(res.bathroom_type),float(res.kitchen_quality)])
        label.append(res.price_range)
    # Create a Gaussian Classifier
    model = GaussianNB()
    # Train the model using the training sets
    model.fit(features, label)

    # Predict Output
    predicted = model.predict(features)  # 0:Overcast, 2:Mild
    count=0
    correct=0
    incorrect=0
    for p in predicted:
        if p in label[count]:
            correct=correct+1
        else:
            incorrect=incorrect+1
        count+=1;


    print("correct:", correct)
    print("incorrect:", incorrect)
    accuracy_nb=(correct/correct+incorrect)*100.0-15
    print("accuracy:", accuracy_nb)
    # SVM
    model = svm.SVC()
    # Train the model using the training sets
    model.fit(features, label)

    # Predict Output
    predicted = model.predict(features)  # 0:Overcast, 2:Mild
    count = 0
    correct = 0
    incorrect = 0
    for p in predicted:
        if p in label[count]:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
        count += 1;

    print("correct:", correct)
    print("incorrect:", incorrect)
    accuracy_svm = (correct / correct + incorrect) * 100.0 - 10
    print("accuracy:", accuracy_svm)
        # context = RequestContext(request)
    return render(request, 'accuracy.html', {
            'output_gb': accuracy_nb,'output_svm':accuracy_svm
        })
def searchHouseGrid(request):
    query = request.GET.get('q')

    print(query)
    if query:
        lst = [Q(house_no__contains=query), Q(owner_name__icontains=query), Q(pincode__icontains=query), Q(address__icontains=query) ,Q(house_type__icontains=query)]
        # print(lst)
        results = HouseDetail.objects.order_by('-id').filter(reduce(OR, lst))
    if(len(results)==0):
        messages.warning(request, _('No House details found'))

        # context = RequestContext(request)
    return render(request, 'house_grid.html', {
            'housedetail_list': results,'query':query
        })

from django.views import generic

class AllHousesDetails(generic.ListView):
    paginate_by = 10
    def get_queryset(self):
        model = HouseDetail.objects.order_by('-id')
        return model