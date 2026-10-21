import os
import sys
import django
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "course_rec_app.settings")
django.setup()

from sklearn.metrics import mean_squared_error, mean_absolute_error
from recommender.models import Interaction, Course
from recommender.collaborative import get_collaborative_recommendations
from recommender.content_based import get_content_based_recommendations
from recommender.hybrid import get_hybrid_recommendations

def evaluate_all(n=10, k=20):
    
    print("--- Starting Full Evaluation ---")


from recommender.collaborative import funk_svd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_all(n=10, k=10):
    print("--- Starting Full Evaluation (Offline) ---")

    interactions = Interaction.objects.all().values('user__id', 'course__original_id', 'rating')
    df = pd.DataFrame(list(interactions))
    
    courses = Course.objects.all().values('original_id', 'title', 'description', 'concepts')
    df_courses = pd.DataFrame(list(courses))
    df_courses.set_index('original_id', inplace=True)
    
    if df.empty:
        print("No interactions found.")
        return

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    print("Training CF (FunkSVD) on Train set...")
    
    user_ids = train_df['user__id'].unique()
    item_ids = train_df['course__original_id'].unique()
    
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {c: i for i, c in enumerate(item_ids)}
    idx_to_item = {i: c for c, i in item_to_idx.items()}
    
    train_df_encoded = train_df.copy()
    train_df_encoded['user_idx'] = train_df_encoded['user__id'].map(user_to_idx)
    train_df_encoded['item_idx'] = train_df_encoded['course__original_id'].map(item_to_idx)
    
    P, Q = funk_svd(train_df_encoded[['user_idx', 'item_idx', 'rating']], n_epochs=20)
    
    print("Building CBF Matrices...")
    df_courses['soup'] = df_courses['title'] + " " + df_courses['description'] + " " + df_courses['concepts']
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_courses['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    course_id_to_idx = {cid: i for i, cid in enumerate(df_courses.index)}

    def get_cbf_score(user_id, target_course_id):
        if target_course_id not in course_id_to_idx: return 0.0
        target_idx = course_id_to_idx[target_course_id]
        
        user_history = train_df[(train_df['user__id'] == user_id) & (train_df['rating'] >= 3.0)]
        liked_courses = user_history['course__original_id'].values
        
        if len(liked_courses) == 0: return 0.0
        
        sim_sum = 0
        count = 0
        for liked in liked_courses:
            if liked in course_id_to_idx:
                l_idx = course_id_to_idx[liked]
                sim_sum += cosine_sim[target_idx, l_idx]
                count += 1
        return sim_sum / count if count > 0 else 0.0

    results = {
        'CF': {'rmse': [], 'mae': [], 'precision': [], 'recall': [], 'f1': []},
        'CBF': {'precision': [], 'recall': [], 'f1': []},
        'Hybrid': {'precision': [], 'recall': [], 'f1': []}
    }
    
    test_users = test_df['user__id'].unique()
    
    all_items = list(item_ids) 
    
    print(f"Evaluating on {len(test_users)} users...")
    
    y_true_all = []
    y_pred_cf_all = []
    
    for user_id in test_users:
        if user_id not in user_to_idx: continue 
        
        u_idx = user_to_idx[user_id]
        
        user_test_ratings = test_df[test_df['user__id'] == user_id]
        
        for _, row in user_test_ratings.iterrows():
            cid = row['course__original_id']
            y_true_all.append(row['rating'])
            
            if cid in item_to_idx:
                i_idx = item_to_idx[cid]
                pred = np.dot(P[u_idx], Q[i_idx])
                pred = max(1.0, min(5.0, pred))
            else:
                pred = 3.0 
            y_pred_cf_all.append(pred)

        relevant_items = set(user_test_ratings[user_test_ratings['rating'] >= 3.0]['course__original_id'])
        if not relevant_items: continue
        
        seen_in_train = set(train_df[train_df['user__id'] == user_id]['course__original_id'])
        candidates = [c for c in all_items if c not in seen_in_train]
        
        cf_scores = []
        cbf_scores = []
        
        for cid in candidates:
            if cid in item_to_idx:
                i_idx = item_to_idx[cid]
                s_cf = np.dot(P[u_idx], Q[i_idx])
            else:
                s_cf = 0 
            cf_scores.append((cid, s_cf))
            
            s_cbf = get_cbf_score(user_id, cid)
            cbf_scores.append((cid, s_cbf))
            
        cf_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_cf = [x[0] for x in cf_scores[:k]]
        
        cbf_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_cbf = [x[0] for x in cbf_scores[:k]]
        
        hybrid_scores = []
        cbf_dict_lookup = dict(cbf_scores)
        for cid, val_cf in cf_scores:
            val_cbf = cbf_dict_lookup.get(cid, 0)
            norm_cf = (val_cf - 1) / 4.0 if val_cf > 1 else 0
            score_h = 0.5 * norm_cf + 0.5 * val_cbf
            hybrid_scores.append((cid, score_h))
            
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_hybrid = [x[0] for x in hybrid_scores[:k]]
        
        def calc_metrics(top_k_list, rel_set):
            hits = len(set(top_k_list) & rel_set)
            prec = hits / k
            rec = hits / len(rel_set)
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            return prec, rec, f1
            
        p, r, f = calc_metrics(top_k_cf, relevant_items)
        results['CF']['precision'].append(p)
        results['CF']['recall'].append(r)
        results['CF']['f1'].append(f)
        
        p, r, f = calc_metrics(top_k_cbf, relevant_items)
        results['CBF']['precision'].append(p)
        results['CBF']['recall'].append(r)
        results['CBF']['f1'].append(f)
        
        p, r, f = calc_metrics(top_k_hybrid, relevant_items)
        results['Hybrid']['precision'].append(p)
        results['Hybrid']['recall'].append(r)
        results['Hybrid']['f1'].append(f)
        
    results['CF']['rmse'] = np.sqrt(mean_squared_error(y_true_all, y_pred_cf_all))
    results['CF']['mae'] = mean_absolute_error(y_true_all, y_pred_cf_all)
    
    def avg(l): return np.mean(l) if l else 0.0
    
    print("\n--- Full Evaluation Results ---")
    print(f"{'Metric':<10} | {'CF':<10} | {'CBF':<10} | {'Hybrid':<10}")
    print("-"*45)
    print(f"{'RMSE':<10} | {results['CF']['rmse']:.4f} | {'N/A':<10} | {'N/A':<10}")
    print(f"{'MAE':<10} | {results['CF']['mae']:.4f} | {'N/A':<10} | {'N/A':<10}")
    print(f"{'Precision@K':<10} | {avg(results['CF']['precision']):.4f} | {avg(results['CBF']['precision']):.4f} | {avg(results['Hybrid']['precision']):.4f}")
    print(f"{'Recall@K':<10} | {avg(results['CF']['recall']):.4f} | {avg(results['CBF']['recall']):.4f} | {avg(results['Hybrid']['recall']):.4f}")
    print(f"{'F1@K':<10} | {avg(results['CF']['f1']):.4f} | {avg(results['CBF']['f1']):.4f} | {avg(results['Hybrid']['f1']):.4f}")
    print("-"*45)

if __name__ == "__main__":
    evaluate_all(n=10, k=5)
