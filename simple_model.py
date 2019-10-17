import numpy as np
import scipy
from scipy.special import expit as sigmoid
from scipy import stats
import random

# from optim import *

def sgd(w, X, y, lr_rate, loss_type="cross_entropy"):
    """
    Function to do stochastic gradient descent
    Based on the loss type, it will calculate the gradient on randomly chosen
    one sample point and return the updated weight w

    Inputs:
    - w: weight vector, shape (D, )
    - X: input data, shape (N, D)
    - y: true labels/values of input data, shape (N, )
    - loss_type: type of loss function used, default is cross entropy loss

    Returns:
    - w: updated weight vector, shape (D, )
    """
    i = np.random.randint(0, X.shape[0])
    if loss_type == "cross_entropy":
        try:
            grad = - lr_rate * (y[i] - self.z[i]) * X[i]
        except:
            z = sigmoid(np.dot(X, w))
            grad = - lr_rate * (y[i] - z[i]) * X[i]
    w -= grad
    return w

class LogisticRegression:
    """
    LogisticRegression calculates the cross entropy loss and performs stochastic
    gradient descent on the given training data, which are all defined in this
    same simple_model.py.

    The LogisticRegression can be used for both training and test/validation dataset.
    It has both fit() method and predict() method as well as accuracy_score() method.

    To train a model you will need to create a LogisticRegression instance, setting some
    hyperparameters such as penalty type, and convergence radius. Then, you will
    call the fit() method passing in training data and labels. This will do
    stocahstic gradient descent and find the optimal weight w, and will store in
    self.w.

    When predicting, it will use the optimized weight self.w and calculate
    sigmoid(np.dot(X, self.w)) which is the "probability" being in class 1. Threshold
    of 0.5 is used to determine whether sample is in class 1 or not and will return
    an array of predicted labels.

    Example:

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    """

    def __init__(self, **kwargs):
        """
        Constructs a new LogisticRegression instance.

        Optional arguments:
        - lr_rate: int, learning rate when doing gradient descent. Default is 0.1
        - lr_decay: int, learning rate decay rate. Values range (0, 1]. Default is 1 (don't decay)
        - penalty: string, "l1" or "l2", defines the penalty term. "l1" will penalize
          weight with l1 norm, and "l2" will penalize weight with l2 norm. Coefficient of
          penalization term is defined with argument C. If set "none"/None, there will be
          no penalization term. Default is None
        - C: int, coefficient for penalization term. Default is 1
        - conv_rad: int, convergence radius for when doing gradient descent. When
          the difference between two consecutive losses are smaller than
          conv_rad, the loop will terminate and return self.w at that point as the
          optimal w. Default is 1e-4
        # - update_rule: A string giving the name of an update rule in this file.
        #   Default is 'sgd'.
        - verbose: Boolean, if set to false then no output will be printed during
          training. Default is True
        - max_iter: int, number of maximum iterations. Default is 100
        """
        # Unpack keyword arguments
        self.lr_rate = kwargs.pop("lr_rate", 0.1)
        self.lr_decay = kwargs.pop("lr_decay", 1)
        self.penalty = kwargs.pop("penalty", None)
        self.C = kwargs.pop("C", 1)
        self.conv_rad = kwargs.pop("conv_rad", 1e-4)
        # self.update_rule = kwargs.pop("update_rule", "sgd")
        self.verbose = kwargs.pop("verbose", True)
        self.max_iter = kwargs.pop("max_iter", 100)

        self.w = None
        self.loss = 0
        self.loss_diff = 1e5

    def cross_entropy_loss(self, y):
        """
        Calculates the cross entropy loss for y (true labels/values) and z (predicted
        labels/values)
        """
        N = y.shape[0]
        loss = 0
        for i in range(N):
            loss -= y[i] * np.log(self.z[i]) + (1 - y[i]) * np.log(self.z[i])

        return loss

    def _step(self, X, y):
        """
        Function defining one gradient descent step
        Called in fit() method and should not be called individually
        """
        self.z = sigmoid(np.dot(X, self.w))
        next_loss = self.cross_entropy_loss(y)
        if self.penalty is not None or self.penalty == "none":
            if self.penalty == "l1":
                next_loss += self.C * np.sum(np.abs(self.w))
            if self.penalty == "l2":
                next_loss += self.C * np.sum(self.w ** 2)
        self.loss_diff = np.abs(self.loss - next_loss)
        self.loss = next_loss
        self.w = sgd(self.w, X, y, self.lr_rate)
        self.lr_rate *= self.lr_decay

    def fit(self, X, y):
        """
        Run optimization to train the model
        """
        self.w = np.zeros((X.shape[1], ))
        for _ in range(self.max_iter):
            self._step(X, y)
            if self.loss_diff < self.conv_rad or np.isnan(self.loss):
                break
            if self.verbose:
                print(self.loss)
                # print(self.w)

    def predict(self, X):
        """
        Run prediction based on trained model
        """
        if self.w is None:
            print("Weight w is not yet optimized. Call fit() first.")
        pred_proba = sigmoid(np.dot(X, self.w))
        y_pred = pred_proba > 0.5
        return y_pred

    def predict_proba(self, X):
        """
        Instead of returning predicted labels, returns predicted probabilities
        """
        pred_proba = sigmoid(np.dot(X, self.w))
        return pred_proba

    def accuracy_score(self, y, y_pred):
        """
        Calculates the accuracy score
        """
        assert len(y) == len(y_pred)
        score = 0
        for i in range(len(y)):
            if y[i] == y_pred[i]:
                score += 1
        score /= len(y)

        return score

"""
DecisionTree algorithm:
 X: sample pts, n samples, d features
 y: labels, binary

-For each feature = feat
 (if numerical)
 -try splitting by midpoint of each two samples
  -thresh = midpoint value
  -calculate h_before
  -left = sample pts with X[feat] < thresh
  -right = sample pts with X[feat] > thresh
 -calculate information gain (ig)
  -calculate h_left and h_right
  -calculate ig -> append to ig_list
 -get max(ig_list) -> append to igmax_list
-get max(igmax_list)
-split where max(igmax_list)
"""

class DecisionTree:

    def __init__(self, tree_size, print_rule=False):
        self.tree_size = tree_size
        self.feat = None
        self.thresh = None
        self.is_leaf = False
        self.l_child = None
        self.r_child = None
        self.leaf_label = 0
        self.print_rule = print_rule

    def entropy(self, y):
        n = len(y)
        uniq, cnt = np.unique(y, return_counts=True)
        h = - np.sum([(cnt[i] / n) * np.log2(cnt[i] / n) for i in range(len(cnt))])
        return h

    def informationGain(self, X, y, col, thresh):
        h_before = self.entropy(y)
        _, _, y_left, y_right = self.split(X, y, col=col, thresh=thresh)
        h_after_l = self.entropy(y_left)
        h_after_r = self.entropy(y_right)
        h_after = (len(y_left) * h_after_l + len(y_right) * h_after_r) / (len(y_left) + len(y_right))
        ig = h_before - h_after

        return ig

    def split(self, X, y, col, thresh):
        l_indx = np.where(X[:, col] <= thresh)[0]
        r_indx = np.where(X[:, col] > thresh)[0]

        X_left = X[l_indx]
        X_right = X[r_indx]
        y_left = y[l_indx]
        y_right = y[r_indx]
        return X_left, X_right, y_left, y_right

    def segmenter(self, X, y, random_=False):
        ig_list = []
        max_ig = 0
        seg_feat = "__"
        seg_thresh = 0

        cols = np.arange(0, X.shape[1]).tolist()
        if random_:
            choose_num = int(np.round(np.sqrt(X.shape[1]))) * 2
            cols = random.sample(cols, choose_num)

        for col in cols:
            uniq = np.unique(X[:, col])
            best_ig = 0
            best_thresh = 0
            for i in range(1, len(uniq)):
                thresh = (uniq[i - 1] + uniq[i]) / 2
                ig = self.informationGain(X, y, col, thresh)
                if ig > best_ig:
                    best_ig = ig
                    best_thresh = thresh
            ig_list.append(best_ig)
            if best_ig > max_ig:
                seg_feat = np.argmax(ig_list)
                seg_thresh = best_thresh
                max_ig = best_ig

        if self.print_rule:
            print("feat {}, thresh {}".format(seg_feat, seg_thresh))
        if max_ig > 0:
            X_left, X_right, y_left, y_right = self.split(X, y, col=seg_feat, thresh=seg_thresh)
            X_child = [X_left, X_right]
            y_child = [y_left, y_right]
            return X_child, y_child, seg_feat, seg_thresh
        else:
            return X, y, seg_feat, seg_thresh

    def fit(self, X, y):
        if self.tree_size == 0:
            self.is_leaf = True
            if len(y) != 0:
                self.leaf_label = stats.mode(y)[0][0]
        if self.is_leaf == False:
            if len(np.unique(y)) == 1:
                self.is_leaf = True
                self.leaf_label = stats.mode(y)[0][0]
            else:
                X_node, y_node, self.feat, self.thresh = self.segmenter(X, y)
                if np.any(np.asarray([len(i) if type(i) != np.int64 else i for i in y_node]) == 0) or self.feat == "__":
                    self.is_leaf = True
                    self.leaf_label = stats.mode(y)[0][0]
                else:
                    self.l_child = DecisionTree(tree_size=self.tree_size - 1, print_rule=self.print_rule)
                    self.l_child.fit(X_node[0], y_node[0])

                    self.r_child = DecisionTree(tree_size=self.tree_size - 1, print_rule=self.print_rule)
                    self.r_child.fit(X_node[1], y_node[1])

    def predict(self, X):
        if self.print_rule:
            print("feat {}, thresh {}".format(self.feat, self.thresh))
        # need to run fit before
        if self.is_leaf == True:
            return self.leaf_label
        else:
            if X[self.feat] < self.thresh:
                if self.print_rule:
                        print("left")
                return self.l_child.predict(X)
            else:
                if self.print_rule:
                        print("right")
                return self.r_child.predict(X)


class RandomForest():

    def __init__(self, tree_size):
        self.tree_size = tree_size
        self.feat = None
        self.thresh = None
        self.is_leaf = False
        self.l_child = None
        self.r_child = None
        self.leaf_label = 0

    def fit(self, X, y):
        if self.tree_size == 0:
            self.is_leaf = True
            if len(y) != 0:
                self.leaf_label = stats.mode(y)[0][0]
        if self.is_leaf == False:
            if len(np.unique(y)) == 1:
                self.is_leaf = True
                self.leaf_label = stats.mode(y)[0][0]
            else:
                tree = DecisionTree(tree_size=self.tree_size)
                X_node, y_node, self.feat, self.thresh = tree.segmenter(X, y, random_=True)
                if np.any(np.asarray([len(i) for i in y_node]) == 0) or self.feat == "__":
                    self.is_leaf = True
                    self.leaf_label = stats.mode(y)[0][0]
                else:
                    self.l_child = RandomForest(tree_size=self.tree_size - 1)
                    self.l_child.fit(X_node[0], y_node[0])

                    self.r_child = RandomForest(tree_size=self.tree_size - 1)
                    self.r_child.fit(X_node[1], y_node[1])

    def predict(self, X):
        if self.is_leaf == True:
            return self.leaf_label
        else:
            if X[self.feat] < self.thresh:
                return self.l_child.predict(X)
            else:
                return self.r_child.predict(X)
