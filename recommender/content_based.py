import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .models import Course, Interaction
from django.db.models import Q, Count


_CBF_CACHE = {
    "tfidf_matrix": None,
    "tfidf_model": None,
    "id_to_idx": None,
    "idx_to_id": None,
    "n_courses": 0,
    "timestamp": 0
}


def _build_cbf_cache():
    global _CBF_CACHE
    courses = list(Course.objects.all())
    n_courses = len(courses)

    if _CBF_CACHE['tfidf_matrix'] is not None and n_courses == _CBF_CACHE['n_courses']:
        return _CBF_CACHE

    course_data = []
    id_to_idx = {}
    idx_to_id = {}

    for idx, c in enumerate(courses):
        soup = f"{c.title} {c.description} {c.concepts}"
        course_data.append(soup)
        id_to_idx[c.id] = idx
        idx_to_id[idx] = c.id

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(course_data)

    _CBF_CACHE.update({
        "tfidf_matrix": tfidf_matrix,
        "tfidf_model": tfidf,
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "n_courses": n_courses,
        "timestamp": pd.Timestamp.now()
    })

    return _CBF_CACHE


def get_content_based_recommendations(user_id, n=5):
    
    cache = _build_cbf_cache()

    tfidf_matrix = cache['tfidf_matrix']
    id_to_idx = cache['id_to_idx']
    idx_to_id = cache['idx_to_id']
    courses = list(Course.objects.all())

    user_interactions = Interaction.objects.filter(
        user_id=user_id
    ).filter(Q(rating__gte=3.0) | Q(rating=0.0))

    liked_course_ids = [i.course.id for i in user_interactions]

    if not liked_course_ids:
        popular_courses = (
            Interaction.objects.values('course__id')
            .annotate(num_interactions=Count('id'))
            .order_by('-num_interactions')[:n]
        )
        recommendations = []
        for c in popular_courses:
            try:
                recommendations.append(Course.objects.get(id=c['course__id']))
            except Course.DoesNotExist:
                continue
        return recommendations

    sim_scores_total = {}

    for liked_id in liked_course_ids:
        if liked_id not in id_to_idx:
            continue
        idx = id_to_idx[liked_id]
        cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
        related_indices = cosine_sim.argsort()[:-(10+2):-1]

        for i in related_indices:
            if i == idx:
                continue
            c_id = idx_to_id[i]
            if c_id in liked_course_ids:
                continue
            sim_scores_total[c_id] = sim_scores_total.get(c_id, 0) + cosine_sim[i]

    sorted_scores = sorted(sim_scores_total.items(), key=lambda x: x[1], reverse=True)
    top_ids = [x[0] for x in sorted_scores[:n]]

    recommended_courses = [Course.objects.get(id=tid) for tid in top_ids]
    return recommended_courses


def get_similar_courses(course_id, n=5):
    
    cache = _build_cbf_cache()
    tfidf_matrix = cache['tfidf_matrix']
    id_to_idx = cache['id_to_idx']
    idx_to_id = cache['idx_to_id']
    courses = list(Course.objects.all())

    if course_id not in id_to_idx:
        return []

    idx = id_to_idx[course_id]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    related_indices = cosine_sim.argsort()[:-(n+2):-1]

    recommendations = []
    for i in related_indices:
        if i != idx:
            recommendations.append(courses[i])

    return recommendations[:n]


def get_content_based_similarities(course_db_id):
    
    cache = _build_cbf_cache()
    tfidf_matrix = cache['tfidf_matrix']
    id_to_idx = cache['id_to_idx']
    idx_to_id = cache['idx_to_id']

    if course_db_id not in id_to_idx:
        return {}

    idx = id_to_idx[course_db_id]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = {idx_to_id[i]: score for i, score in enumerate(cosine_sim) if i != idx}
    return sim_scores
