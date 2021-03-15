#<-------- Imports -------->#
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from CV import CV
from Regression import OLS, TLS, RANSAC


#<-------- Polynomial Regression Class -------->#
class PolyRegression:

    def __init__(self, x, y, q, predict_type):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.q = q
        self.predict_type = predict_type
    
    # Generates the X matrix from x,y datapoints  
    def gen_x_matrix(self):      
        len_x = len(self.x)

        self.X = np.zeros((len_x, self.q+1))
        for row in range(len_x):
            for col in range(self.q+1):
                self.X[row, col] = self.x[row]**col

    # Calculates the weight vector using passed in least squared solver
    def get_weight_vector(self):
        self.gen_x_matrix()
        self.w, self.eqn_str = self.predict_type.train(self.X, self.y)

    # Predicts f(x) based on calculated w
    def predict(self, x_test):
        inner_lam = lambda x_i: np.asarray([x_i**i for i in range(len(self.w))])
        g_x = [np.dot(self.w, inner_lam(x_i)) for x_i in x_test]
        g_x = np.transpose(g_x)
        return g_x

# Runs cv, estimation, and generates plots
def get_plot(vid_name):
    print("--------------------------------------")
    print(vid_name[3:])

    # Run CV for given video
    comp_vis = CV(vid_name)

    img = comp_vis.run_cv()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Estimate trajectory for the bottom of the ball
    all_data_bottom = np.asarray(comp_vis.bottom_ball)
    x_bottom_data = all_data_bottom[:,0]
    y_bottom_data = all_data_bottom[:,1]

    # Ordinary least squares (OLS)
    bottom_w_OLS = PolyRegression(x_bottom_data, y_bottom_data, 2, OLS)
    bottom_w_OLS.get_weight_vector()

    # Total least squares (TLS)
    bottom_w_TLS = PolyRegression(x_bottom_data, y_bottom_data, 2, TLS)
    bottom_w_TLS.get_weight_vector()

    # RANSAC
    bottom_w_RANSAC = PolyRegression(x_bottom_data, y_bottom_data, 2, RANSAC)
    bottom_w_RANSAC.get_weight_vector()

    # Estimate trajectory for the bottom of the ball
    all_data_top = np.asarray(comp_vis.top_ball)
    x_top_data = all_data_top[:,0]
    y_top_data = all_data_top[:,1]

    # OLS
    top_w_OLS = PolyRegression(x_top_data, y_top_data, 2, OLS)
    top_w_OLS.get_weight_vector()

    # TLS
    top_w_TLS = PolyRegression(x_top_data, y_top_data, 2, TLS)
    top_w_TLS.get_weight_vector()

    # RANSAC
    top_w_RANSAC = PolyRegression(x_top_data, y_top_data, 2, RANSAC)
    top_w_RANSAC.get_weight_vector()

    # Plot results
    plt.imshow(img)
    plt.plot(x_top_data, top_w_OLS.predict(x_top_data), label=f'OLS, {top_w_OLS.eqn_str}')
    plt.plot(x_top_data, top_w_TLS.predict(x_top_data), label=f'TLS, {top_w_TLS.eqn_str}')
    plt.plot(x_top_data, top_w_RANSAC.predict(x_top_data), label=f'RANSAC, {top_w_RANSAC.eqn_str}')
    
    plt.legend()
    plt.title(f"Top Trajectory\n{vid_name[3:]}")
    plt.grid(True)

    plt.figure()
    plt.imshow(img)
    plt.plot(x_bottom_data, bottom_w_OLS.predict(x_bottom_data), label=f'OLS, {bottom_w_OLS.eqn_str}')
    plt.plot(x_bottom_data, bottom_w_TLS.predict(x_bottom_data), label=f'TLS, {bottom_w_TLS.eqn_str}')
    plt.plot(x_bottom_data, bottom_w_RANSAC.predict(x_bottom_data), label=f'RANSAC, {bottom_w_RANSAC.eqn_str}')

    plt.legend()
    plt.title(f"Bottom Trajectory\n{vid_name[3:]}")
    plt.grid(True)
    plt.show()

    print("")

#<-------- Main -------->#
if __name__ == '__main__':
    # Clear terminal
    # os.system('cls')

    # Run all
    get_plot('../Ball_travel_10fps.mp4') 
    get_plot('../Ball_travel_2_updated.mp4') 

