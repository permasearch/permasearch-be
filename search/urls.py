from django.urls import path
from .views import *
app_name = "search"

urlpatterns = [
  path("docs/", get_document, name='search'),
  path("", get_serp, name='serp'),
  # path("index", do_index ,name='index')
]