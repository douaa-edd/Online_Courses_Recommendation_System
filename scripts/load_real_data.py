import os
import sys
import pandas as pd
import numpy as np
import django

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "course_rec_app.settings")
django.setup()

from django.contrib.auth.models import User
from recommender.models import Course, Interaction

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def run():
    print("Starting OULAD Module-Based Ingestion...")
    
    print("Clearing existing database...")
    Interaction.objects.all().delete()
    Course.objects.all().delete()
    User.objects.exclude(is_superuser=True).delete()
    
    print("Reading courses.csv...")
    courses_df = pd.read_csv(os.path.join(DATA_DIR, 'courses.csv'))
    
    print(f"Creating {len(courses_df)} Modules (Courses)...")
    course_batch = []
    
    
    CODE_TO_NAME = {
        'AAA': 'Social Sciences',
        'BBB': 'Psychology & Health',
        'CCC': 'Computer Science',
        'DDD': 'STEM',
        'EEE': 'General Science',
        'FFF': 'Mathematics & Stats',
        'GGG': 'Humanities'
    }

    for _, row in courses_df.iterrows():
        code_mod = row['code_module']
        code_pres = row['code_presentation']
        
        readable_name = CODE_TO_NAME.get(code_mod, code_mod)
        title = f"{readable_name} ({code_pres})" 
        
        length = row.get('module_presentation_length', 'N/A')
        description = f"Official Open University Module: {readable_name}. Code: {code_mod}, Presentation: {code_pres}. Duration: {length} days."
        
        c = Course(
            original_id=f"{code_mod}_{code_pres}",
            title=title,
            description=description,
            school="Open University",
            teacher=f"Faculty of {readable_name}",
            concepts=f"{readable_name}, {code_mod}, {code_pres}"
        )
        course_batch.append(c)
        
    Course.objects.bulk_create(course_batch)
    

    
    print("Indexing courses...")
    all_courses = Course.objects.all()
    course_map = {} 
    for c in all_courses:
        parts = c.original_id.split('_')
        if len(parts) == 2:
             key = (parts[0], parts[1])
             course_map[key] = c
             
    print(f"Indexed {len(course_map)} courses.")

    
    CHUNK_SIZE = 1000000 
    print(f"Reading interactions (limit {CHUNK_SIZE} rows)...")
    
    interactions_df = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'), nrows=CHUNK_SIZE, 
                                  usecols=['code_module', 'code_presentation', 'id_student', 'sum_click'])
    
    print("Aggregating engagement per Student-Module...")
    grouped = interactions_df.groupby(['code_module', 'code_presentation', 'id_student'])['sum_click'].sum().reset_index()
    
    print(f"Found {len(grouped)} unique Student-Module pairs.")
    
    grouped['log_clicks'] = np.log1p(grouped['sum_click'])
    max_log = grouped['log_clicks'].max()
    min_log = grouped['log_clicks'].min()
    
    if max_log == min_log:
        grouped['rating'] = 5.0
    else:
        grouped['rating'] = 1 + 4 * ((grouped['log_clicks'] - min_log) / (max_log - min_log))
        
    grouped['rating'] = grouped['rating'].round(1)
    
    unique_student_ids = grouped['id_student'].unique()
    print(f"Creating {len(unique_student_ids)} Users...")
    
    default_pass_hash = 'pbkdf2_sha256$720000$TjYdxME+dyF3$G/a4O/QOQv1fI0leIxzcUUREA='
    
    existing_users = set(User.objects.values_list('username', flat=True))
    user_batch = []
    
    for sid in unique_student_ids:
        s_username = str(sid)
        if s_username not in existing_users:
            u = User(username=s_username, password=default_pass_hash)
            user_batch.append(u)
            
    if user_batch:
        User.objects.bulk_create(user_batch, ignore_conflicts=True)
        print(f"Inserted {len(user_batch)} users.")
        
    print("Mapping users...")
    user_map = {u.username: u for u in User.objects.all()}
    
    print("Creating database interactions...")
    interaction_batch = []
    
    records = grouped.to_dict('records')
    
    for row in records:
        c_mod = row['code_module']
        c_pres = row['code_presentation']
        sid = str(row['id_student'])
        rating = row['rating']
        
        user_obj = user_map.get(sid)
        course_obj = course_map.get((c_mod, c_pres))
        
        if user_obj and course_obj:
            interaction_batch.append(Interaction(
                user=user_obj,
                course=course_obj,
                rating=rating
            ))
            
        if len(interaction_batch) >= 5000:
            Interaction.objects.bulk_create(interaction_batch, ignore_conflicts=True)
            interaction_batch = []
            
    if interaction_batch:
        Interaction.objects.bulk_create(interaction_batch, ignore_conflicts=True)


    print("="*30)
    print("FINAL STATS (Reverted to Modules):")
    print(f"Courses: {Course.objects.count()}")
    print(f"Users:   {User.objects.count()}")
    print(f"Ratings: {Interaction.objects.count()}")
    print("="*30)

if __name__ == "__main__":
    run()
