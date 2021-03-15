#<-------- Imports -------->#
import numpy as np
from math import log, ceil

"""
Least squares solvers
"""

#<-------- Ordinary Least Squares -------->#
class OLS:
    def __init__(self):
        pass

    # Train model
    @staticmethod
    def train(X, y, show_w_flag = True):
        X__T = np.transpose(X)
        XT_X = X__T @ X
        XT_X_inv = np.linalg.pinv(XT_X)
        w = (XT_X_inv @ X__T) @ np.transpose(y)
        eqn_str = f"$y={round(w[-1], 5)}x^2{round(w[-2],2)}x+{round(w[-3],2)}$"
        if show_w_flag:
            print("\nUsing OLS")
            print(eqn_str)
        return w, eqn_str

#<-------- Total Least Squares -------->#
class TLS:
    def __init__(self):
        pass

    # Train model
    @staticmethod
    def train(X, y):
        Xy = np.hstack((X, np.c_[y]))
        X__T = np.transpose(X)
        XT_X = X__T @ X
        eigs,_ = np.linalg.eig(Xy.T @ Xy)
        eigs.sort()
        eigs = eigs[::-1]
        n = XT_X.shape[0]

        x_tls = np.linalg.pinv(XT_X - eigs[-1]*np.eye(n)) @ X__T @ y

        w = x_tls
        eqn_str = f"$y={round(w[-1], 5)}x^2{round(w[-2],2)}x+{round(w[-3],2)}$"
        print("\nUsing TLS")
        print(eqn_str)

        return x_tls, eqn_str

#<-------- RANSAC -------->
class RANSAC:
    def __init__(self):
        pass

    # Train model
    @staticmethod
    def train(X, y, e=.7, s=3, p=.99, t=15):
        Xy = np.hstack((X, np.c_[y]))
        random_list = range(X.shape[0])
        sample_count = 0
        N = ceil(log(1-p)/log(1-(1-e)**s))

        eval_line = lambda x, w: np.dot(x, w)
        all_pts = lambda X, y, w: [1 if y_i > eval_line(x_row, w) else -1 for x_row, y_i in zip(X, y)]

        max_inlier_count = 0
        final_w = []

        while sample_count < N:
            # Select 3 random points
            r_pt1 = np.random.choice(random_list)
            r_pt2 = np.random.choice(random_list)
            r_pt3 = np.random.choice(random_list)

            X_pt1 = Xy[r_pt1, :]
            X_pt2 = Xy[r_pt2, :]
            X_pt3 = Xy[r_pt3, :]

            X_train = np.vstack((X_pt1, X_pt2, X_pt3))

            # Compute line using ordinary least squares
            ols = OLS()
            w,_ = ols.train(X_train[:, 0:-1], X_train[:,-1], show_w_flag=False)
            
            # Determine upper and lower thresholds
            w_upper = w.copy()
            w_upper[0] += t

            w_lower = w.copy()
            w_lower[0] -= t
            
            # Calculates datapoints relative position to each line. 1 if datapoint is above line and -1 if below line 
            above_upper_line = all_pts(X, y, w_upper)
            above_lower_line = all_pts(X, y, w_lower)

            # For each point, multiply the corresponding point in for two lists above. -1 means above lower line and 
            # below upper line. Therefore, the data point must be an inlier
            inlier_count = 0
            total_count = len(above_lower_line)
            for point in range(total_count):
                if above_upper_line[point]*above_lower_line[point] == -1:
                    inlier_count += 1

            # Check if line gives max inliers. If so, that's our final weight vector
            if inlier_count > max_inlier_count:
                max_inlier_count = inlier_count
                final_w = w

            sample_count += 1

        eqn_str = f"$y={round(final_w[-1], 5)}x^2{round(final_w[-2],2)}x+{round(final_w[-3],2)}$"
        print("\nUsing RANSAC")
        print(eqn_str)
        return final_w, eqn_str