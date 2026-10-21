from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from .models import Course, Interaction
from .content_based import get_similar_courses, get_content_based_recommendations
from .collaborative import get_collaborative_recommendations
from .hybrid import get_hybrid_recommendations

@login_required
def course_detail(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    user_rating = None
    is_enrolled = False
    
    if request.user.is_authenticated:
        try:
            interaction = Interaction.objects.get(user=request.user, course=course)
            user_rating = interaction.rating
            is_enrolled = True
        except Interaction.DoesNotExist:
            user_rating = None
            is_enrolled = False
            
    if request.method == "POST":
        rating = float(request.POST.get('rating', 0))
        Interaction.objects.update_or_create(
            user=request.user,
            course=course,
            defaults={'rating': rating}
        )
        return redirect('course_detail', course_id=course.id)
    
    cbf_recs = get_similar_courses(course.id, n=4)

    return render(request, 'recommender/course_detail.html', {
        'course': course, 
        'user_rating': user_rating,
        'cbf_recs': cbf_recs,
        'is_enrolled': is_enrolled
    })
from .hybrid import get_hybrid_recommendations

def index(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'recommender/index.html')

def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def dashboard(request):
    user_id = request.user.id
    
    cf_recs = get_collaborative_recommendations(user_id, n=6)
    
    hybrid_recs = get_hybrid_recommendations(user_id, n=6)
    
    cbf_recs = get_content_based_recommendations(user_id, n=6)
    
    is_cold_start = False
    if not cf_recs:
        is_cold_start = True
        cf_recs = Course.objects.all().order_by('?')[:6]
        
    context = {
        'cf_recs': cf_recs,
        'hybrid_recs': hybrid_recs,
        'cbf_recs': cbf_recs,
        'is_cold_start': is_cold_start
    }
    return render(request, 'recommender/dashboard.html', context)

@login_required
def course_list(request):
    courses = Course.objects.all().order_by('original_id')
    return render(request, 'recommender/course_list.html', {'courses': courses})

@login_required
def course_detail(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    user_rating = None
    is_enrolled = False
    
    if request.user.is_authenticated:
        try:
            interaction = Interaction.objects.get(user=request.user, course=course)
            user_rating = interaction.rating
            is_enrolled = True
        except Interaction.DoesNotExist:
            user_rating = None
            is_enrolled = False
            
    if request.method == "POST":
        rating = float(request.POST.get('rating', 0))
        Interaction.objects.update_or_create(
            user=request.user,
            course=course,
            defaults={'rating': rating}
        )
        return redirect('course_detail', course_id=course.id)
    
    cbf_recs = get_similar_courses(course.id, n=4)

    return render(request, 'recommender/course_detail.html', {
        'course': course, 
        'user_rating': user_rating,
        'cbf_recs': cbf_recs,
        'is_enrolled': is_enrolled
    })

@login_required
def enroll_course(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    Interaction.objects.get_or_create(user=request.user, course=course, defaults={'rating': 0.0})
    return redirect('course_content', course_id=course.id)

@login_required
def course_content(request, course_id):
    course = get_object_or_404(Course, id=course_id)
    interaction = Interaction.objects.filter(user=request.user, course=course).first()
    if not interaction:
        return redirect('course_detail', course_id=course.id)
        
    user_rating = interaction.rating
        
    if request.method == "POST":
        rating = float(request.POST.get('rating', 0))
        interaction.rating = rating
        interaction.save()
        return redirect('course_content', course_id=course.id)
        
    return render(request, 'recommender/course_content.html', {
        'course': course,
        'user_rating': user_rating
    })

@login_required
def my_courses(request):
    interactions = Interaction.objects.filter(user=request.user).select_related('course')
    return render(request, 'recommender/my_courses.html', {'interactions': interactions})

@login_required
def profile(request):
    return render(request, 'recommender/profile.html', {'user': request.user})


