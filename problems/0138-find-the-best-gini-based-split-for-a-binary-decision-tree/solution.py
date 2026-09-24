import torch
from typing import Tuple

def find_best_split(X: torch.Tensor, y: torch.Tensor) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    def cal_gini_impurity(y:torch.Tensor):
        if len(y)==0:return 0
        p=y.float().mean()
        return 1-(p**2+(1-p)**2)
    n_samples,n_features=X.shape
    best_feature,best_thr,best_g=0,0.0,torch.inf
    for feature_idx in range(n_features):
        column=X[:,feature_idx]
        for thr in torch.unique(column):
            left_mask=column<=thr
            right_mask=~left_mask
            if left_mask.sum()==0 or right_mask.sum()==0:continue
            weight_g=(left_mask.sum()*cal_gini_impurity(y[left_mask])+right_mask.sum()*cal_gini_impurity(y[right_mask]))/n_samples
            if weight_g<best_g:
                best_g=weight_g
                best_feature=feature_idx
                best_thr=thr.item()
    return best_feature,best_thr
