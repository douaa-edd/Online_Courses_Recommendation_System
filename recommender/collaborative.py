import numpy as np
import pandas as pd
from .models import Interaction, Course
import time


_MODEL_CACHE = None
_LAST_INTERACTIONS_COUNT = 0


def funk_svd(ratings_df, n_factors=10, n_epochs=50, lr=0.005, reg=0.02):

    n_users = ratings_df['user_idx'].nunique()
    n_items = ratings_df['item_idx'].nunique()

    P = np.random.normal(scale=0.1, size=(n_users, n_factors))
    Q = np.random.normal(scale=0.1, size=(n_items, n_factors))

    for _ in range(n_epochs):
        for row in ratings_df.itertuples():
            u = row.user_idx
            i = row.item_idx
            r_ui = row.rating

            pred = np.dot(P[u], Q[i])
            err = r_ui - pred

            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P[u] - reg * Q[i])

    return P, Q


def get_trained_model():
    
    global _MODEL_CACHE, _LAST_INTERACTIONS_COUNT

    interactions = Interaction.objects.all()
    n_interactions = interactions.count()

    if _MODEL_CACHE is not None and n_interactions == _LAST_INTERACTIONS_COUNT:
        return _MODEL_CACHE

    print("Training Funk-SVD model...")

    df = pd.DataFrame(list(interactions.values('user__id', 'course__original_id', 'rating')))
    if df.empty:
        return None

    user_ids = df['user__id'].unique()
    item_ids = df['course__original_id'].unique()
    user_to_idx = {u: idx for idx, u in enumerate(user_ids)}
    item_to_idx = {i: idx for idx, i in enumerate(item_ids)}
    idx_to_item = {idx: i for i, idx in item_to_idx.items()}

    df['user_idx'] = df['user__id'].map(user_to_idx)
    df['item_idx'] = df['course__original_id'].map(item_to_idx)

    P, Q = funk_svd(
        df[['user_idx', 'item_idx', 'rating']],
        n_factors=10,
        n_epochs=10,
        lr=0.007,
        reg=0.02
    )

    _MODEL_CACHE = {
        'P': P,
        'Q': Q,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'df': df
    }
    _LAST_INTERACTIONS_COUNT = n_interactions

    return _MODEL_CACHE


def update_user_interaction(user_id, item_id, rating, n_epochs=5, lr=0.007, reg=0.02):
   
    model = get_trained_model()
    if model is None:
        return

    df = model['df']
    P = model['P']
    Q = model['Q']
    user_to_idx = model['user_to_idx']
    item_to_idx = model['item_to_idx']
    idx_to_item = model['idx_to_item']

    if user_id not in user_to_idx:
        new_idx = len(user_to_idx)
        user_to_idx[user_id] = new_idx
        P = np.vstack([P, np.random.normal(scale=0.1, size=(1, P.shape[1]))])

    if item_id not in item_to_idx:
        new_idx = len(item_to_idx)
        item_to_idx[item_id] = new_idx
        idx_to_item[new_idx] = item_id
        Q = np.vstack([Q, np.random.normal(scale=0.1, size=(1, Q.shape[1]))])

    u_idx = user_to_idx[user_id]
    i_idx = item_to_idx[item_id]

    for _ in range(n_epochs):
        pred = np.dot(P[u_idx], Q[i_idx])
        err = rating - pred
        P[u_idx] += lr * (err * Q[i_idx] - reg * P[u_idx])
        Q[i_idx] += lr * (err * P[u_idx] - reg * Q[i_idx])

    model['P'] = P
    model['Q'] = Q
    model['user_to_idx'] = user_to_idx
    model['item_to_idx'] = item_to_idx
    model['idx_to_item'] = idx_to_item

    df = pd.concat([df, pd.DataFrame({'user__id':[user_id],'course__original_id':[item_id],'rating':[rating],'user_idx':[u_idx],'item_idx':[i_idx]})], ignore_index=True)
    model['df'] = df
    global _MODEL_CACHE
    _MODEL_CACHE = model


def get_collaborative_recommendations(user_id, n=5):
   
    model = get_trained_model()
    if model is None:
        return Course.objects.all().order_by('?')[:n]

    P = model['P']
    Q = model['Q']
    user_to_idx = model['user_to_idx']
    idx_to_item = model['idx_to_item']
    df = model['df']

    if user_id not in user_to_idx:
        popular_courses = (
            df.groupby('course__original_id')
            .size()
            .sort_values(ascending=False)
            .head(n)
            .index.tolist()
        )
        recommendations = []
        for cid in popular_courses:
            try:
                recommendations.append(Course.objects.get(original_id=cid))
            except Course.DoesNotExist:
                continue
        return recommendations

    u_idx = user_to_idx[user_id]
    scores = np.dot(Q, P[u_idx])
    item_ids_ordered = [idx_to_item[i] for i in range(len(scores))]
    score_series = pd.Series(scores, index=item_ids_ordered)

    rated_items = df[df['user__id'] == user_id]['course__original_id'].unique()
    score_series = score_series.drop(rated_items, errors='ignore')

    top_ids = score_series.sort_values(ascending=False).head(n).index.tolist()
    recommendations = []
    for cid in top_ids:
        try:
            recommendations.append(Course.objects.get(original_id=cid))
        except Course.DoesNotExist:
            continue

    return recommendations
