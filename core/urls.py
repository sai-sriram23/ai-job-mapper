from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('api/parse-resume/', views.parse_resume_api, name='parse_resume_api'),
    path('api/analyze-career/', views.analyze_career_api, name='analyze_career_api'),
    path('api/why-what/', views.why_what_api, name='why_what_api'),
    path('api/market-insights/', views.market_insights_api, name='market_insights_api'),
    path('api/live-jobs/', views.live_jobs_api, name='live_jobs_api'),

    # Course generator primary endpoints
    path('api/course-outline/', views.course_outline_api, name='course_outline_api'),
    path('api/week-details/', views.week_details_api, name='week_details_api'),
    path('api/day-details/', views.day_details_api, name='day_details_api'),
    path('api/ollama-status/', views.ollama_status_api, name='ollama_status_api'),
    path('api/ollama-chat/', views.ollama_chat_api, name='ollama_chat_api'),

    # ai-course-gen1 direct REST API compatibility aliases
    path('api/models', views.models_api, name='models_api'),
    path('api/health', views.health_api, name='health_api'),
    path('api/generate/outline', views.course_outline_api, name='gen_outline_alias'),
    path('api/generate/week', views.week_details_api, name='gen_week_alias'),
    path('api/generate/day', views.day_details_api, name='gen_day_alias'),
]

