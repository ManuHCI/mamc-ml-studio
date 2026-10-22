"""The model catalogue.

For every algorithm this file holds:
  * the plain-language teaching content (analogy, how it works, what it does to your data,
    strengths, weaknesses, when to use),
  * its settings (hyperparameters) with simple explanations and tuning grids,
  * a factory that builds the scikit-learn estimator.
"""
from __future__ import annotations

import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

_LOGREG_PENALTY_DEPRECATED = inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"


class LogisticModel(ClassifierMixin, BaseEstimator):
    """Logistic regression with a simple 'penalty' switch that works on old and new scikit-learn."""

    def __init__(self, C=1.0, penalty="l2", class_weight=None, max_iter=5000, random_state=None):
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    def _make(self):
        kw = dict(C=self.C, class_weight=self.class_weight, max_iter=self.max_iter, random_state=self.random_state,
                  solver="saga" if self.penalty == "l1" else "lbfgs")
        if _LOGREG_PENALTY_DEPRECATED:
            kw["l1_ratio"] = 1.0 if self.penalty == "l1" else 0.0
        else:
            kw["penalty"] = self.penalty
        return LogisticRegression(**kw)

    def fit(self, X, y, sample_weight=None):
        self.model_ = self._make().fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)


class SVMModel(ClassifierMixin, BaseEstimator):
    """SVC with probability estimates (Platt scaling) that works on old and new scikit-learn."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", class_weight=None, random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        from sklearn.calibration import CalibratedClassifierCV
        svc = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, class_weight=self.class_weight,
                  random_state=self.random_state)
        _, counts = np.unique(y, return_counts=True)
        folds = int(max(2, min(5, counts.min())))
        self.model_ = CalibratedClassifierCV(svc, method="sigmoid", cv=folds, ensemble=False)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def _hp(name, label, kind, default, explain, low="", high="", options=None, rng=None, grid=None, fmt=None, advanced=False):
    """kind: int | float | logfloat | choice"""
    return dict(name=name, label=label, kind=kind, default=default, explain=explain, low=low, high=high,
                options=options, range=rng, grid=grid, fmt=fmt, advanced=advanced)


MODELS = {
    # ------------------------------------------------------------------ linear
    "logreg": dict(
        name="Logistic Regression", tasks=["classification"], family="Linear", scale=True, balance="class_weight",
        one_line="Gives every factor a weight, adds them up into a risk score, and converts the score into a probability.",
        analogy="Exactly like a clinical risk score (Framingham, CHA2DS2-VASc): each risk factor adds or subtracts points, "
                "and the total points tell you the chance of disease.",
        how=["Starts with every factor's weight at zero.",
             "Looks at the training patients and adjusts the weights so that patients who had the outcome get high scores "
             "and those who did not get low scores.",
             "Passes the score through an S-shaped curve (the logistic or sigmoid function) so it becomes a probability between 0 and 1.",
             "A patient is called 'positive' when the probability crosses a cut-off (0.5 by default; you can change it)."],
        data_effect="Draws ONE straight dividing line (a flat surface when there are many factors) through your patients. "
                    "Everyone on one side gets a higher probability. It cannot bend the line around curved patterns unless you "
                    "create such factors yourself.",
        strengths=["Easy to explain: weights become odds ratios", "Works well with small datasets", "Fast; rarely overfits when regularised",
                   "Gives well-calibrated probabilities"],
        weaknesses=["Misses curved (non-linear) relationships and interactions", "Sensitive to extreme values",
                    "Needs factors on similar scales (the software scales them for you)"],
        when="First choice for most medical prediction questions, especially with fewer than a few hundred patients or "
             "when you must explain the model. Always try it as the baseline.",
        example="Predicting 5-year diabetes risk from age, BMI and fasting glucose.",
        params=[
            _hp("C", "Regularisation C (trust in the data)", "logfloat", 1.0,
                "How much the model may trust the training patients. Small C keeps the weights small and cautious "
                "(simpler model, less overfitting). Large C lets the weights grow to fit the training data closely.",
                "Very cautious, simple model", "Fits training data closely - may overfit", rng=(0.001, 100.0),
                grid=[0.01, 0.1, 1.0, 10.0, 100.0]),
            _hp("penalty", "Type of regularisation", "choice", "l2",
                "L2 (ridge) shrinks all weights a little. L1 (lasso) can switch unhelpful factors off completely (weight = 0) - "
                "an automatic way of choosing variables.", options=["l2", "l1"], grid=["l2", "l1"],
                fmt={"l2": "L2 - ridge (shrink all)", "l1": "L1 - lasso (drop weak factors)"}),
        ]),
    "linreg": dict(
        name="Linear Regression", tasks=["regression"], family="Linear", scale=True, balance=None,
        one_line="Fits the best straight line (or flat surface) through the data to predict a number.",
        analogy="Like the classic formula 'predicted value = baseline + a x factor1 + b x factor2 ...' - e.g. a formula that "
                "estimates eGFR from creatinine, age and sex.",
        how=["Finds the line that keeps the total of squared prediction errors as small as possible (least squares).",
             "Each factor gets a coefficient: how much the prediction changes when that factor goes up by one unit.",
             "Prediction for a new patient = intercept + sum of (coefficient x value)."],
        data_effect="Assumes every factor has a straight-line effect that simply adds up. Curves and interactions are ignored.",
        strengths=["Very easy to explain", "Needs little data", "No settings to tune", "Coefficients are interpretable"],
        weaknesses=["Misses curved relationships", "Sensitive to extreme values", "Can be unstable when factors are highly correlated"],
        when="The baseline for any 'predict a number' question.",
        example="Predicting length of hospital stay from age, CRP and albumin.",
        params=[]),
    "ridge": dict(
        name="Ridge Regression", tasks=["regression"], family="Linear", scale=True, balance=None,
        one_line="Linear regression with a brake that stops coefficients from becoming too large.",
        analogy="A cautious senior who says 'do not over-interpret any single factor' - all effects are shrunk a little.",
        how=["Same as linear regression, but adds a penalty for large coefficients.",
             "The penalty strength (alpha) decides how much the coefficients are shrunk towards zero."],
        data_effect="Still a straight-line model, but more stable when factors are many or correlated with each other.",
        strengths=["More stable than plain linear regression", "Handles correlated factors", "Fast"],
        weaknesses=["Still misses curves and interactions"],
        when="When you have many factors or factors that measure similar things (weight, BMI, waist).",
        example="Predicting systolic BP from many correlated anthropometric measures.",
        params=[_hp("alpha", "Penalty strength (alpha)", "logfloat", 1.0,
                    "How hard the coefficients are pulled towards zero. 0 = plain linear regression.",
                    "Almost plain linear regression", "Strong shrinkage - very simple model", rng=(0.001, 100.0),
                    grid=[0.01, 0.1, 1.0, 10.0, 100.0])]),
    "lasso": dict(
        name="Lasso Regression", tasks=["regression"], family="Linear", scale=True, balance=None,
        one_line="Linear regression that can switch off unhelpful factors completely.",
        analogy="A strict examiner who keeps only the factors that really matter and gives the rest zero marks.",
        how=["Same as linear regression, plus an L1 penalty on the size of the coefficients.",
             "With enough penalty, weak factors get a coefficient of exactly zero - they drop out of the model."],
        data_effect="A straight-line model that also selects variables for you.",
        strengths=["Automatic variable selection", "Simple final model"],
        weaknesses=["When two factors are correlated it may keep one at random", "Misses curves"],
        when="Many candidate factors and you want a short list of the important ones.",
        example="Choosing which of 20 lab values predict ICU stay.",
        params=[_hp("alpha", "Penalty strength (alpha)", "logfloat", 0.1,
                    "How strongly weak factors are pushed to zero. Larger alpha = fewer factors kept.",
                    "Keeps almost all factors", "Keeps only the strongest factors", rng=(0.0001, 10.0),
                    grid=[0.001, 0.01, 0.1, 1.0])]),
    # ------------------------------------------------------------------ trees
    "tree": dict(
        name="Decision Tree", tasks=["classification", "regression"], family="Tree", scale=False, balance="class_weight",
        one_line="Learns a flowchart of yes/no questions that ends in a prediction.",
        analogy="A clinical algorithm or triage flowchart: 'Is MCV < 80? -> Is ferritin < 15? -> Iron deficiency'.",
        how=["Looks at every factor and every possible cut-off, and picks the ONE question that best separates the outcome groups.",
             "Splits the patients into two branches by that question.",
             "Repeats the same search inside each branch, again and again.",
             "Stops when the tree reaches its maximum depth or the groups become too small. The last boxes ('leaves') give the prediction."],
        data_effect="Cuts your patients into rectangular boxes with straight cuts along each factor. Each box gets the "
                    "majority answer (or the average value) of the training patients inside it.",
        strengths=["Easiest model to explain - you can draw it", "Captures cut-offs and interactions",
                   "No scaling needed", "Not disturbed by extreme values"],
        weaknesses=["A deep tree memorises the training patients (overfits badly)", "Small changes in the data can change the whole tree",
                    "Usually less accurate than forests or boosting"],
        when="When you want a simple, printable decision rule, or to teach how ML thinks.",
        example="A 3-question rule to tell iron deficiency from thalassaemia trait.",
        params=[
            _hp("max_depth", "Maximum depth (number of questions)", "int", 4,
                "The longest chain of questions from the top of the tree to a leaf. Deeper trees fit the training data "
                "better but memorise noise.", "Very simple rule - may underfit", "Very detailed rule - may overfit",
                rng=(1, 20), grid=[2, 3, 4, 6, 8, None]),
            _hp("min_samples_leaf", "Minimum patients in a leaf", "int", 5,
                "A final box must contain at least this many training patients. Larger values give safer, simpler trees.",
                "Leaves can be single patients (overfit)", "Only large groups (simpler)", rng=(1, 50), grid=[1, 5, 10, 20]),
            _hp("criterion", "How a good question is judged", "choice", "gini",
                "Gini and entropy are two almost identical ways of measuring how 'pure' a group is. Rarely matters.",
                options=["gini", "entropy"], grid=None, advanced=True, fmt={"gini": "Gini impurity", "entropy": "Entropy (information gain)"}),
        ]),
    "rf": dict(
        name="Random Forest", tasks=["classification", "regression"], family="Tree ensemble", scale=False, balance="class_weight",
        one_line="Grows hundreds of different decision trees and lets them vote.",
        analogy="A large board of doctors. Each one has seen a slightly different set of patients and was allowed to look at "
                "only some of the tests. Individually they make mistakes, but the majority vote is very reliable.",
        how=["Makes a 'bootstrap' copy of the training patients (random draws with replacement) for each tree.",
             "Grows a decision tree on each copy; at every question the tree may choose only from a random subset of factors.",
             "Repeats this for hundreds of trees, so the trees are all different.",
             "For a new patient, every tree votes; the forest reports the share of votes (probability) or the average value."],
        data_effect="Averages many box-shaped trees, so the dividing boundary becomes a smooth, flexible shape that can follow "
                    "curves and interactions in your data.",
        strengths=["Accurate on most tabular medical data", "Hard to overfit badly", "Handles curves, cut-offs and interactions",
                   "Little tuning needed; no scaling needed", "Gives a feature-importance ranking"],
        weaknesses=["A 'black box' - no single rule to print", "Larger and slower than a single tree",
                    "Probabilities can be slightly under-confident"],
        when="A strong default when you have a few hundred patients or more and care about accuracy.",
        example="Predicting 30-day readmission from 25 clinical variables.",
        params=[
            _hp("n_estimators", "Number of trees", "int", 200,
                "How many doctors sit on the board. More trees = more stable results but slower. Beyond a few hundred the gain is small.",
                "Few trees - results jump around", "Many trees - stable but slower", rng=(20, 1000), grid=[100, 300]),
            _hp("max_depth", "Maximum depth of each tree", "int", None,
                "How many questions each tree may ask. 'No limit' is usual for forests because voting cancels out the overfitting.",
                "Shallow trees - simpler", "Deep trees - more detail", rng=(1, 30), grid=[3, 5, 8, None]),
            _hp("max_features", "Factors considered at each question", "choice", "sqrt",
                "At each question a tree may only look at a random subset of factors. Smaller subsets make the trees more different "
                "from each other (that is what makes the forest strong).", options=["sqrt", "log2", 0.5, 1.0],
                grid=["sqrt", 0.5], fmt={"sqrt": "Square root of the number of factors", "log2": "log2 of the number of factors",
                                         0.5: "Half of the factors", 1.0: "All factors (= 'bagging')"}),
            _hp("min_samples_leaf", "Minimum patients in a leaf", "int", 1,
                "Smallest group allowed at the end of each branch. Larger = smoother, safer model.", "Very detailed", "Smoother",
                rng=(1, 30), grid=[1, 3, 5, 10]),
        ]),
    "gb": dict(
        name="Gradient Boosting", tasks=["classification", "regression"], family="Tree ensemble", scale=False, balance="class_weight",
        one_line="Builds small trees one after another; each new tree corrects the mistakes of the previous ones.",
        analogy="A relay of residents: the first gives a rough answer, the second concentrates on the cases the first got wrong, "
                "the third fixes what is still wrong ... the team answer keeps improving. (The same idea as XGBoost and LightGBM.)",
        how=["Starts with a simple guess (the average risk).",
             "Grows a small tree that predicts the ERRORS of the current guess.",
             "Adds a small fraction (the learning rate) of that tree to the guess.",
             "Repeats this hundreds of times; each step reduces the remaining error a little."],
        data_effect="Builds a very flexible boundary step by step, concentrating on the patients that are hardest to classify.",
        strengths=["Often the most accurate method for tabular data", "Captures complex patterns and interactions",
                   "Fast (histogram-based implementation)"],
        weaknesses=["More settings to tune", "Can overfit small datasets if too many trees or too high a learning rate",
                    "A black box"],
        when="When accuracy matters most and you have several hundred patients or more.",
        example="Predicting sepsis risk from vital signs and laboratory values.",
        params=[
            _hp("max_iter", "Number of trees (boosting rounds)", "int", 200,
                "How many correction steps are made. Too few = underfit; too many (with a high learning rate) = overfit.",
                "Few corrections", "Many corrections", rng=(20, 1000), grid=[100, 200, 400]),
            _hp("learning_rate", "Learning rate (step size)", "logfloat", 0.05,
                "How much of each new tree is added. Small steps learn slowly but carefully (need more trees); big steps learn fast but may overshoot.",
                "Slow, careful learning", "Fast, risky learning", rng=(0.005, 1.0), grid=[0.02, 0.05, 0.1, 0.2]),
            _hp("max_depth", "Depth of each small tree", "int", 3,
                "How many questions each small tree asks. 2-4 is usual for boosting.", "Very simple trees", "Complex trees",
                rng=(1, 10), grid=[2, 3, 5]),
            _hp("min_samples_leaf", "Minimum patients in a leaf", "int", 10,
                "Smallest group allowed at the end of a branch.", "Detailed", "Smooth", rng=(1, 50), grid=[5, 10, 20]),
            _hp("l2_regularization", "L2 regularisation", "float", 0.0,
                "An extra brake on how strongly each leaf can push the prediction.", "No brake", "Strong brake",
                rng=(0.0, 10.0), grid=None, advanced=True),
        ]),
    # ------------------------------------------------------------------ others
    "svm": dict(
        name="Support Vector Machine", tasks=["classification", "regression"], family="Kernel", scale=True, balance="class_weight",
        one_line="Draws the widest possible 'street' between the groups of patients.",
        analogy="Imagine two groups of patients on a map. The SVM draws the boundary road that keeps the widest safety margin "
                "from both groups. With the 'kernel trick' it can bend the road around curved patterns.",
        how=["Places every patient as a point in a space with one axis per factor.",
             "Finds the boundary with the largest margin from the nearest patients of each group (the 'support vectors').",
             "With the RBF kernel it measures how similar patients are to each other, which allows curved boundaries.",
             "The C setting decides how many patients may be on the wrong side of the street."],
        data_effect="Separates your patients with a smooth boundary that can be straight (linear kernel) or curved (RBF kernel), "
                    "decided mainly by the borderline patients.",
        strengths=["Works well with many factors and modest sample sizes", "Can capture curved boundaries",
                   "Resistant to overfitting with sensible C"],
        weaknesses=["Slow on large datasets (> 5,000 patients)", "Black box", "Needs scaling and tuning (C, gamma)",
                    "Probabilities are estimated indirectly"],
        when="Medium-sized datasets with many continuous factors, e.g. lab panels or imaging features.",
        example="Classifying tumours as malignant or benign from 10 cell-nucleus measurements.",
        params=[
            _hp("C", "C (how strictly errors are punished)", "logfloat", 1.0,
                "Small C allows some patients on the wrong side for a wider, simpler street. Large C tries to classify every "
                "training patient correctly (narrow street, risk of overfitting).", "Wide street - simple", "Narrow street - may overfit",
                rng=(0.01, 100.0), grid=[0.1, 1.0, 10.0, 100.0]),
            _hp("kernel", "Kernel (shape of the boundary)", "choice", "rbf",
                "Linear = a straight boundary. RBF = a flexible, curved boundary. Poly = a polynomial curve.",
                options=["rbf", "linear", "poly"], grid=["rbf", "linear"],
                fmt={"rbf": "RBF - curved (most common)", "linear": "Linear - straight", "poly": "Polynomial"}),
            _hp("gamma", "Gamma (reach of each patient)", "choice", "scale",
                "For curved kernels: how far the influence of one patient reaches. Large gamma = very local, wiggly boundary "
                "(overfit); small gamma = smooth boundary.", options=["scale", 0.001, 0.01, 0.1, 1.0],
                grid=["scale", 0.01, 0.1], fmt={"scale": "Automatic ('scale')"}),
        ]),
    "knn": dict(
        name="K-Nearest Neighbours", tasks=["classification", "regression"], family="Similarity", scale=True, balance=None,
        one_line="Finds the k most similar previous patients and copies their majority answer.",
        analogy="An experienced clinician thinking 'this patient reminds me of the last 7 similar patients I saw - 5 of them had dengue'.",
        how=["Stores all the training patients (there is no real 'training').",
             "For a new patient, measures the distance (difference in every factor) to every stored patient.",
             "Picks the k nearest ones.",
             "Predicts the majority outcome among them (or their average value for numbers)."],
        data_effect="Gives each region of your data the answer of the nearby patients. Small k follows every individual patient "
                    "(jagged boundary); large k smooths it out.",
        strengths=["Very intuitive", "No assumptions about the shape of the data", "Naturally handles many groups"],
        weaknesses=["Must be scaled (otherwise a factor in big units dominates the distance)", "Slow with large datasets",
                    "Poor with many irrelevant factors", "Cannot say which factors matter"],
        when="Small-to-medium datasets with a few meaningful continuous factors; great for teaching.",
        example="Suggesting the type of anaemia from the 6 red-cell indices of the most similar previous haemograms.",
        params=[
            _hp("n_neighbors", "k - number of neighbours", "int", 7,
                "How many similar patients vote. k = 1 copies the single most similar patient (overfits); a very large k "
                "ignores local patterns (underfits). Odd numbers avoid ties.", "Follows every patient - overfits",
                "Very smooth - may underfit", rng=(1, 60), grid=[3, 5, 7, 11, 15, 25]),
            _hp("weights", "Vote weighting", "choice", "uniform",
                "Uniform = every neighbour has one vote. Distance = closer neighbours have bigger votes.",
                options=["uniform", "distance"], grid=["uniform", "distance"],
                fmt={"uniform": "Equal votes", "distance": "Closer = stronger vote"}),
            _hp("p", "Distance measure", "choice", 2,
                "Euclidean = straight-line distance. Manhattan = sum of the differences in each factor.",
                options=[2, 1], grid=None, advanced=True, fmt={2: "Euclidean", 1: "Manhattan"}),
        ]),
    "nb": dict(
        name="Naive Bayes", tasks=["classification"], family="Probabilistic", scale=False, balance="sample_weight",
        one_line="Bayes' theorem: starts from the pre-test probability and updates it with each finding.",
        analogy="Exactly how clinicians use likelihood ratios: pre-test probability x LR of test 1 x LR of test 2 ... = post-test "
                "probability. 'Naive' because it assumes the tests are independent of each other.",
        how=["Learns how common each outcome is (the prior / pre-test probability).",
             "For each outcome group, learns the typical value and spread of every factor (a bell curve for numbers).",
             "For a new patient, multiplies the prior by how typical each finding is in each group.",
             "The group with the highest resulting (posterior) probability is the prediction."],
        data_effect="Describes each outcome group by bell-shaped curves and asks which group's curves fit the new patient best. "
                    "Boundaries are smooth curves.",
        strengths=["Very fast; works with very small datasets", "Easy to connect to clinical Bayesian reasoning",
                   "Naturally handles many groups"],
        weaknesses=["The independence assumption is usually false (e.g. weight and BMI)", "Probabilities are often too extreme",
                    "Usually less accurate than other methods"],
        when="Tiny datasets, quick baselines, and for teaching Bayesian thinking.",
        example="A first screen for the type of anaemia from red-cell indices.",
        params=[_hp("var_smoothing", "Variance smoothing", "logfloat", 1e-9,
                    "A tiny safety margin added to every spread so that one unusual value cannot make the model 100% certain.",
                    "Almost no smoothing", "Heavy smoothing", rng=(1e-12, 1e-1), grid=[1e-9, 1e-6, 1e-3, 1e-1], advanced=False)]),
    "mlp": dict(
        name="Neural Network", tasks=["classification", "regression"], family="Neural network", scale=True, balance="sample_weight",
        one_line="Layers of small calculating units ('neurons') that together learn complex patterns.",
        analogy="Loosely like the nervous system: each artificial neuron adds up its weighted inputs and 'fires' if the total is "
                "high enough; layers of neurons build up from simple to complex features. Deep learning uses many such layers.",
        how=["Input layer: one neuron per factor.",
             "Hidden layer(s): each neuron combines all inputs with its own weights and applies a switch (activation).",
             "Output layer: gives the probability (or the number).",
             "Training: shows the training patients many times (epochs); after each pass the weights are nudged in the "
             "direction that reduces the error (back-propagation + gradient descent)."],
        data_effect="Can bend the boundary into almost any shape. With few patients it easily memorises them instead of learning.",
        strengths=["Very flexible - can learn complex patterns", "The basis of image, ECG and text AI"],
        weaknesses=["Needs many patients (usually thousands)", "Black box", "Sensitive to settings and scaling",
                    "Results vary between runs"],
        when="Large datasets. For typical thesis-sized tables (< 1,000 patients) other methods usually do as well or better.",
        example="Predicting mortality from thousands of ICU records.",
        params=[
            _hp("hidden_layer_sizes", "Hidden layers (neurons)", "choice", (32,),
                "How many hidden layers and how many neurons in each. More neurons = more flexible but needs more data.",
                options=[(8,), (16,), (32,), (64,), (32, 16), (64, 32)], grid=[(16,), (32,), (32, 16)],
                fmt={(8,): "1 layer x 8 neurons", (16,): "1 layer x 16 neurons", (32,): "1 layer x 32 neurons",
                     (64,): "1 layer x 64 neurons", (32, 16): "2 layers: 32 + 16 neurons", (64, 32): "2 layers: 64 + 32 neurons"}),
            _hp("alpha", "Regularisation (alpha)", "logfloat", 1e-3,
                "A brake on the size of the weights. Larger alpha = simpler network, less overfitting.", "Free weights", "Strong brake",
                rng=(1e-6, 10.0), grid=[1e-4, 1e-2, 1.0]),
            _hp("learning_rate_init", "Learning rate", "logfloat", 1e-3,
                "How big a step the weights take after each pass. Too small = slow; too large = unstable.", "Small careful steps",
                "Large jumpy steps", rng=(1e-5, 0.5), grid=[1e-3, 1e-2]),
            _hp("max_iter", "Maximum passes over the data (epochs)", "int", 600,
                "How many times the network may study the training patients.", "Short training", "Long training",
                rng=(50, 3000), grid=None, advanced=True),
        ]),
}

# models that we never offer for a given task
TASK_MODELS = {
    "classification": ["logreg", "tree", "rf", "gb", "svm", "knn", "nb", "mlp"],
    "regression": ["linreg", "ridge", "lasso", "tree", "rf", "gb", "svm", "knn", "mlp"],
}


def model_name(key: str, task: str) -> str:
    n = MODELS[key]["name"]
    if key == "svm" and task == "regression":
        return "Support Vector Regression"
    if key in ("tree", "rf", "gb", "knn", "mlp") and task == "regression":
        return n + " (regression)"
    return n


def default_params(key: str) -> dict:
    return {p["name"]: p["default"] for p in MODELS[key]["params"]}


def param_grid(key: str, n_rows: int | None = None) -> dict:
    grid = {}
    for p in MODELS[key]["params"]:
        if p["grid"]:
            g = list(p["grid"])
            if p["name"] == "n_neighbors" and n_rows:
                g = [k for k in g if k < max(n_rows * 0.6, 4)] or [3]
            grid[p["name"]] = g
    return grid


def random_distributions(key: str, n_rows: int | None = None) -> dict:
    """Distributions for random search (lists or scipy distributions)."""
    from scipy.stats import loguniform, randint, uniform
    d = {}
    for p in MODELS[key]["params"]:
        if p["kind"] == "logfloat":
            lo, hi = p["range"]
            d[p["name"]] = loguniform(lo, hi)
        elif p["kind"] == "int":
            lo, hi = p["range"]
            if p["name"] == "n_neighbors" and n_rows:
                hi = int(max(min(hi, n_rows * 0.6), 3))
            vals = list(range(int(lo), int(hi) + 1))
            if p["default"] is None or (p["grid"] and None in p["grid"]):
                d[p["name"]] = vals + [None]
            else:
                d[p["name"]] = randint(int(lo), int(hi) + 1)
        elif p["kind"] == "float":
            lo, hi = p["range"]
            d[p["name"]] = uniform(lo, hi - lo)
        elif p["kind"] == "choice" and not p["advanced"]:
            d[p["name"]] = list(p["options"])
    return d


def make_estimator(key: str, task: str, params: dict | None = None, seed: int = 42, balance: bool = False):
    params = dict(default_params(key), **(params or {}))
    cls = task == "classification"
    cw = "balanced" if (balance and cls and MODELS[key]["balance"] == "class_weight") else None
    if key == "logreg":
        return LogisticModel(C=params["C"], penalty=params["penalty"], class_weight=cw, random_state=seed)
    if key == "linreg":
        return LinearRegression()
    if key == "ridge":
        return Ridge(alpha=params["alpha"], random_state=seed)
    if key == "lasso":
        return Lasso(alpha=params["alpha"], max_iter=20000, random_state=seed)
    if key == "tree":
        if cls:
            return DecisionTreeClassifier(max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                                          criterion=params["criterion"], class_weight=cw, random_state=seed)
        crit = {"gini": "squared_error", "entropy": "squared_error"}.get(params["criterion"], params["criterion"])
        return DecisionTreeRegressor(max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
                                     criterion=crit, random_state=seed)
    if key == "rf":
        kw = dict(n_estimators=params["n_estimators"], max_depth=params["max_depth"], max_features=params["max_features"],
                  min_samples_leaf=params["min_samples_leaf"], random_state=seed, n_jobs=-1)
        return RandomForestClassifier(class_weight=cw, **kw) if cls else RandomForestRegressor(**kw)
    if key == "gb":
        kw = dict(max_iter=params["max_iter"], learning_rate=params["learning_rate"], max_depth=params["max_depth"],
                  min_samples_leaf=params["min_samples_leaf"], l2_regularization=params["l2_regularization"],
                  early_stopping=False, random_state=seed)
        return HistGradientBoostingClassifier(class_weight=cw, **kw) if cls else HistGradientBoostingRegressor(**kw)
    if key == "svm":
        if cls:
            return SVMModel(C=params["C"], kernel=params["kernel"], gamma=params["gamma"], class_weight=cw, random_state=seed)
        return SVR(C=params["C"], kernel=params["kernel"], gamma=params["gamma"])
    if key == "knn":
        kw = dict(n_neighbors=params["n_neighbors"], weights=params["weights"], p=params["p"])
        return KNeighborsClassifier(**kw) if cls else KNeighborsRegressor(**kw)
    if key == "nb":
        return GaussianNB(var_smoothing=params["var_smoothing"])
    if key == "mlp":
        kw = dict(hidden_layer_sizes=tuple(params["hidden_layer_sizes"]), alpha=params["alpha"],
                  learning_rate_init=params["learning_rate_init"], max_iter=params["max_iter"], random_state=seed)
        return MLPClassifier(**kw) if cls else MLPRegressor(**kw)
    raise KeyError(key)


def supports_sample_weight(estimator) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def fmt_value(p: dict, v) -> str:
    if p.get("fmt") and v in p["fmt"]:
        return str(p["fmt"][v])
    if v is None:
        return "No limit"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 0.001 or abs(v) >= 10000:
            return f"{v:.0e}"
        return f"{v:g}"
    return str(v)


def describe_params(key: str, params: dict) -> list[str]:
    """Plain-language sentences for a set of settings."""
    out = []
    pmap = {p["name"]: p for p in MODELS[key]["params"]}
    for k, v in params.items():
        p = pmap.get(k)
        if p:
            out.append(f"{p['label']}: {fmt_value(p, v)}")
    return out
