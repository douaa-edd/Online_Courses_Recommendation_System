from django.db import models
from django.contrib.auth.models import User

class Course(models.Model):
    original_id = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=500)
    description = models.TextField(blank=True, null=True)
    teacher = models.CharField(max_length=500, blank=True, null=True)
    school = models.CharField(max_length=500, blank=True, null=True)
    concepts = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.title

class Interaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='interactions')
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='interactions')
    rating = models.FloatField(default=0.0)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'course')
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username} -> {self.course.title}"
