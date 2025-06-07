import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, recall_score

class outlier_contraction_expansion_classifier:
    def __init__(self, m_neighbor=1, lambda_weight=0.5, threshold_type='contraction', rate=0.1):
        if m_neighbor < 1:
            self.m_neighbor = 1
        else:
            self.m_neighbor = m_neighbor

        self.lambda_weight = lambda_weight 

        if not (0 <= rate < 1): 
            raise ValueError("Rate must be in the range [0, 1).")
        self.rate = rate
        self.threshold_type = threshold_type

        self.class_specific_deviation_thresholds = {} 
        self.characteristics = None
        self.data_train = None
        self.train_cat_feature_arrays_ = {} 
        self.category_attribute = 'target'
        self.independent_attributes_list = []

    def _calculate_internal_distances(self, vector_array):
        n = len(vector_array)
        if n == 0: return None, []
        distances_mat = cdist(vector_array, vector_array, metric='euclidean')
        mask = ~np.eye(n, dtype=bool) 
        per_row_distances_to_others = [distances_mat[i][mask[i]] if n > 1 else np.array([]) for i in range(n)]
        return distances_mat, per_row_distances_to_others

    def vector_category_agg(self, vector_obs, category_data_array, aggregation_type='mean'):
        vector_obs_2d = np.array(list(vector_obs))[np.newaxis, :] 
        if len(category_data_array) == 0: return np.inf 
        distances = cdist(vector_obs_2d, category_data_array, metric='euclidean').ravel() 
        if len(distances) == 0: return np.inf 
        if aggregation_type == 'median': return np.nanmedian(distances)
        elif aggregation_type == 'mean': return np.nanmean(distances)
        elif aggregation_type == 'maximum': return np.nanmax(distances)
        elif aggregation_type == 'm_neighbor':
            count = len(distances)
            k = min(self.m_neighbor, count)
            if k == 0 : return np.inf 
            m_closest = np.sort(distances)[:k]
            return np.nanmean(m_closest) if len(m_closest) > 0 else np.inf
        else: raise ValueError(f"Unknown aggregation type: {aggregation_type}")

    def fit(self, X, y):
        if not self.independent_attributes_list: 
            if isinstance(X, pd.DataFrame):
                self.independent_attributes_list = [col for col in X.columns if col != self.category_attribute]
            else:
                self.independent_attributes_list = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=self.independent_attributes_list) if not isinstance(X, pd.DataFrame) else X[self.independent_attributes_list].copy()
        y_series = pd.Series(y, name=self.category_attribute) if not isinstance(y, pd.Series) else y.rename(self.category_attribute)

        self.data_train = pd.concat([X_df.reset_index(drop=True), y_series.reset_index(drop=True)], axis=1)
        self.category_characteristics(self.data_train)

        self.train_cat_feature_arrays_ = {}
        if self.characteristics and self.characteristics[self.category_attribute]:
            grouped_train_data_for_arrays = self.data_train.groupby(self.category_attribute)
            for label in self.characteristics[self.category_attribute]:
                if label in grouped_train_data_for_arrays.groups:
                    self.train_cat_feature_arrays_[label] = grouped_train_data_for_arrays.get_group(label)[self.independent_attributes_list].to_numpy()
                else:
                    self.train_cat_feature_arrays_[label] = np.array([])

        self.class_specific_deviation_thresholds = {} 
        if self.characteristics and self.characteristics[self.category_attribute]:
            scores_per_class = {cat: [] for cat in self.characteristics[self.category_attribute]}
            for i in range(len(self.data_train)):
                obs = self.data_train.iloc[i]
                true_label = obs[self.category_attribute]
                if true_label not in self.characteristics[self.category_attribute]: continue
                try: 
                    cat_idx = self.characteristics[self.category_attribute].index(true_label)
                except ValueError: 
                    continue 
                obs_true_cat_data_np = self.train_cat_feature_arrays_.get(true_label)
                if obs_true_cat_data_np is None or len(obs_true_cat_data_np) == 0: continue
                obs_vec = obs[self.independent_attributes_list].to_numpy()
                stick_obs_own = self.vector_category_agg(obs_vec, obs_true_cat_data_np, 'm_neighbor')
                incl_obs_own = self.vector_category_agg(obs_vec, obs_true_cat_data_np, 'mean')
                if np.isinf(stick_obs_own) or np.isinf(incl_obs_own) or np.isnan(stick_obs_own) or np.isnan(incl_obs_own): continue
                stick_char_own = self.characteristics['empirical_stickiness'][cat_idx]
                bound_char_own = self.characteristics['empirical_boundary'][cat_idx]
                dev_score = self.deviation_score_function(stick_obs_own, incl_obs_own, stick_char_own, bound_char_own, self.lambda_weight)
                scores_per_class[true_label].append(dev_score)

            for label, scores in scores_per_class.items():
                if scores:
                    scores_array = np.array(scores)
                    if self.threshold_type == 'contraction':
                        threshold_percentile = (1 - self.rate) * 100
                        self.class_specific_deviation_thresholds[label] = np.percentile(scores_array, threshold_percentile)
                    elif self.threshold_type == 'expansion':
                        base_score = np.max(scores_array)
                        expansion_component = np.percentile(scores_array, self.rate * 100)
                        self.class_specific_deviation_thresholds[label] = base_score + expansion_component
                    else:
                        raise ValueError(f"Unsupported threshold_type: {self.threshold_type}. Use 'contraction' or 'expansion'.")
                else:
                    self.class_specific_deviation_thresholds[label] = np.inf 
        return self

    def category_characteristics(self, data_df):
        if not self.independent_attributes_list:
            self.independent_attributes_list = [col for col in data_df.columns if col != self.category_attribute]
        unique_cat_labels = sorted(list(data_df[self.category_attribute].unique()))
        grouped_data = data_df.groupby(self.category_attribute)
        processed_labels, stickiness_vals, boundary_vals = [], [], []

        for label in unique_cat_labels:
            if label not in grouped_data.groups: continue
            cat_features = grouped_data.get_group(label)[self.independent_attributes_list].to_numpy()
            n_cat_points = len(cat_features)
            if n_cat_points == 0: continue
            per_point_m_neighbor_dists_in_cat = []
            per_point_mean_dists_to_others_in_cat = []

            if n_cat_points == 1:
                per_point_m_neighbor_dists_in_cat.append(0.0)
                per_point_mean_dists_to_others_in_cat.append(0.0)
            else:
                _, dists_to_others_for_each_point_list = self._calculate_internal_distances(cat_features)
                for point_dists_to_others in dists_to_others_for_each_point_list:
                    if len(point_dists_to_others) == 0:
                        per_point_m_neighbor_dists_in_cat.append(0.0)
                        per_point_mean_dists_to_others_in_cat.append(0.0)
                        continue
                    k_neighbors = min(self.m_neighbor, len(point_dists_to_others))
                    if k_neighbors > 0:
                        closest_k_dists = np.partition(point_dists_to_others, k_neighbors - 1)[:k_neighbors]
                        per_point_m_neighbor_dists_in_cat.append(np.nanmean(closest_k_dists))
                    else:
                        per_point_m_neighbor_dists_in_cat.append(0.0)
                    per_point_mean_dists_to_others_in_cat.append(np.nanmean(point_dists_to_others))

            stickiness = np.nanmedian(per_point_m_neighbor_dists_in_cat) if per_point_m_neighbor_dists_in_cat else 0.0
            boundary = np.nanmedian(per_point_mean_dists_to_others_in_cat) if per_point_mean_dists_to_others_in_cat else 0.0
            stickiness_vals.append(max(1e-4, stickiness if not np.isnan(stickiness) else 1e-4))
            boundary_vals.append(max(1e-4, boundary if not np.isnan(boundary) else 1e-4))
            processed_labels.append(label)

        self.characteristics = {
            self.category_attribute: processed_labels,
            'empirical_stickiness': stickiness_vals,
            'empirical_boundary': boundary_vals
        }

    def deviation_score_function(self, stick_obs, incl_obs, stick_cat_char, bound_cat_char, lambda_w):
        gamma_w = 1 - lambda_w
        stick_obs = np.asarray(stick_obs)
        incl_obs = np.asarray(incl_obs)
        stick_cat_char = np.asarray(stick_cat_char)
        bound_cat_char = np.asarray(bound_cat_char)

        term1 = lambda_w * ((stick_obs - stick_cat_char) / stick_cat_char)
        term2 = gamma_w * ((incl_obs - bound_cat_char) / bound_cat_char)

        term1 = np.nan_to_num(term1, nan=0.0, posinf=np.inf, neginf=-np.inf)
        term2 = np.nan_to_num(term2, nan=0.0, posinf=np.inf, neginf=-np.inf)

        return term1 + term2

    def predict(self, X):
        if self.characteristics is None or not self.characteristics.get(self.category_attribute):
            return [-1] * (len(X) if hasattr(X, '__len__') else 1)

        if not isinstance(X, pd.DataFrame):
            if not self.independent_attributes_list: 
                raise ValueError("Model not fitted or feature names not determined. Call fit() first.")
            X_df = pd.DataFrame(X, columns=self.independent_attributes_list)
        else:
            if not all(col in X.columns for col in self.independent_attributes_list):
                missing_cols = [col for col in self.independent_attributes_list if col not in X.columns]
                raise ValueError(f"Input DataFrame X is missing columns: {missing_cols}. Expected: {self.independent_attributes_list}")
            X_df = X[self.independent_attributes_list].copy()

        return self._model_prediction(X_df)

    def _model_prediction(self, X_test_df):
        model_cat_labels = self.characteristics[self.category_attribute]
        char_stick_all = np.array(self.characteristics['empirical_stickiness'])
        char_bound_all = np.array(self.characteristics['empirical_boundary'])

        predictions = []
        for i in range(len(X_test_df)):
            obs_vector = X_test_df.iloc[i].to_numpy()
            stickiness_list, inclusion_list = [], []
            for label in model_cat_labels:
                cat_features_np = self.train_cat_feature_arrays_.get(label)
                stickiness = self.vector_category_agg(obs_vector, cat_features_np, 'm_neighbor')
                inclusion = self.vector_category_agg(obs_vector, cat_features_np, 'mean')
                stickiness_list.append(stickiness)
                inclusion_list.append(inclusion)

            stick_arr = np.array(stickiness_list)
            incl_arr = np.array(inclusion_list)

            deviation_scores = self.deviation_score_function(
                stick_arr, incl_arr,
                char_stick_all, char_bound_all,
                self.lambda_weight
            )

            valid_candidates = [
                (score, label) for score, label in zip(deviation_scores, model_cat_labels)
                if score <= self.class_specific_deviation_thresholds.get(label, np.inf)
            ]

            if not valid_candidates:
                predictions.append(-1)
            else:
                predictions.append(min(valid_candidates, key=lambda x: x[0])[1])

        return predictions

    def score(self, y_true, y_pred, accuracy_metric='accuracy'):
        y_true_arr, y_pred_arr = np.asarray(y_true), np.asarray(y_pred)

        if accuracy_metric == 'accuracy':
            correct_predictions = np.sum(y_true_arr == y_pred_arr)
            return correct_predictions / len(y_true_arr) if len(y_true_arr) > 0 else 0.0

        inlier_mask_pred = (y_pred_arr != -1)
        y_true_for_pred_inliers = y_true_arr[inlier_mask_pred]
        y_pred_inliers = y_pred_arr[inlier_mask_pred]

        if len(y_pred_inliers) == 0:
            return 0.0

        if accuracy_metric == 'F1 score':
            return f1_score(y_true_for_pred_inliers, y_pred_inliers, average='macro', zero_division=0)
        elif accuracy_metric == 'recall':
            return recall_score(y_true_for_pred_inliers, y_pred_inliers, average='macro', zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {accuracy_metric}. Must be 'accuracy', 'F1 score', or 'recall'")
