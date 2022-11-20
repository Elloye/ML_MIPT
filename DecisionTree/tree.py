import numpy as np
from sklearn.base import BaseEstimator


def gini(y):
    p = np.bincount(y.flatten().astype('int')) / len(y)
    return 1 - np.sum(p ** 2)

def entropy(y):
    EPS = 0.0005
    p = np.bincount(y.flatten().astype('int')) / len(y)
    return - np.sum(p * np.log2(p + EPS))

def variance(y):
    return np.mean((y - np.mean(y)) ** 2)

def mad_median(y):
    return np.mean(np.abs(y - np.median(y)))


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba
    
    def is_leaf(self):
        return self.value is not None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, max_depth=100, min_samples_split=2, criterion_name = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.root = None
    
    def is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False

    def create_split(self, X, thresh):
        left_idx = np.argwhere(X < thresh).flatten()
        right_idx = np.argwhere(X >= thresh).flatten()
        return left_idx, right_idx

    def information_gain(self, X, y, thresh):
        H = self.all_criterions[self.criterion_name][0]
        parent_loss = H(y)
        left_idx, right_idx = self.create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * H(y[left_idx]) + (n_right / n) * H(y[right_idx])
        return parent_loss - child_loss

    def best_split(self, X, y): 
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                score = self.information_gain(X[:, feat], y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self.is_finished(depth):
            if self.all_criterions[self.criterion_name][1]:
                leaf_value = np.argmax(np.bincount(y.flatten().astype('int')))
                
                leaf_probas = np.zeros(self.n_classes)
                probas = np.bincount(y.flatten().astype('int')) / len(y)
                idx = probas.nonzero()
                leaf_probas[idx] = probas[idx]
                
                return Node(value=leaf_value, proba=leaf_probas)
            
            if self.criterion_name == 'variance':
                leaf_value = np.mean(y)
            
            elif self.criterion_name == 'mad_median':
                leaf_value = np.median(y)                            
            
            return Node(value=leaf_value)    
        
        best_feat, best_thresh = self.best_split(X, y)

        left_idx, right_idx = self.create_split(X[:, best_feat], best_thresh)
        left_child = self.build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self.build_tree(X[right_idx, :], y[right_idx], depth + 1)
        
        return Node(best_feat, best_thresh, left_child, right_child)        
    
    def traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value, node.proba
        
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
     
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))       
        self.root = self.build_tree(X, y) 

    def predict(self, X):
        pred = [self.traverse_tree(x, self.root)[0] for x in X]
        return np.array(pred)

    def predict_proba(self, X):
        pred_proba = [self.traverse_tree(x, self.root)[1] for x in X]
        return np.array(pred_proba)