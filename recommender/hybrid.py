import numpy as np
from .models import Interaction, Course
from .collaborative import get_collaborative_recommendations, get_trained_model
from .content_based import get_content_based_recommendations, get_content_based_similarities

def get_cf_scores(user_id):
    model = get_trained_model()
    if model is None or user_id not in model['user_to_idx']:
        return {}

    u_idx = model['user_to_idx'][user_id]
    P = model['P']
    Q = model['Q']
    idx_to_item = model['idx_to_item']

    scores = np.dot(Q, P[u_idx])

    return {
        idx_to_item[i]: scores[i]
        for i in range(len(scores))
    }


def get_cbf_scores(user_id):
    interactions = (
        Interaction.objects
        .filter(user_id=user_id)
        .order_by('-id')
    )

    if not interactions.exists():
        return {}

    last_course = interactions.first().course
    similarities = get_content_based_similarities(last_course.id)

    return similarities 

def get_weighted_hybrid_scores(user_id, alpha=0.5):
   
    cf_scores = get_cf_scores(user_id)
    cbf_scores = get_cbf_scores(user_id)

    all_items = set(cf_scores) | set(cbf_scores)

    hybrid_scores = {}

    for item_id in all_items:
        cf = cf_scores.get(item_id, 0)
        cbf = cbf_scores.get(item_id, 0)
        hybrid_scores[item_id] = alpha * cf + (1 - alpha) * cbf
        
    return hybrid_scores

def get_hybrid_recommendations(user_id, n=5, alpha=0.5):
   
    hybrid_scores = get_weighted_hybrid_scores(user_id, alpha)

    if not hybrid_scores:
        return get_collaborative_recommendations(user_id, n)

    ranked_items = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True
    )[:n]

    recommendations = []
    for cid in ranked_items:
        try:
            recommendations.append(Course.objects.get(original_id=cid))
        except Course.DoesNotExist:
            continue

    return recommendations
