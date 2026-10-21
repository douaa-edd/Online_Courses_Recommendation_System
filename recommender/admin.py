from django.contrib import admin
from .models import Course, Interaction

@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ('title', 'original_id', 'school', 'teacher')
    search_fields = ('title', 'original_id')

@admin.register(Interaction)
class InteractionAdmin(admin.ModelAdmin):
    list_display = ('user', 'course', 'rating', 'timestamp')
    list_filter = ('rating', 'timestamp')
    search_fields = ('user__username', 'course__title')
