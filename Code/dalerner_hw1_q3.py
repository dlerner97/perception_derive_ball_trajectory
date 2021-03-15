#<-------- Imports -------->#
import numpy as np
from os import system
import pandas as pd

#<-------- Main -------->#
if __name__ == '__main__':
    # Clear terminal
    # system('cls')

    # Define A matrix
    pts = {1: {'x':   5, 'y':   5, 'xp': 100, 'yp': 100},
           2: {'x': 150, 'y':   5, 'xp': 200, 'yp':  80},
           3: {'x': 150, 'y': 150, 'xp': 220, 'yp':  80},
           4: {'x':   5, 'y': 150, 'xp': 100, 'yp': 200}}

    A = [[-pts[1]['x'], -pts[1]['y'], -1, 0, 0, 0, pts[1]['x']*pts[1]['xp'], pts[1]['y']*pts[1]['xp'], pts[1]['xp']],
         [0, 0, 0, -pts[1]['x'], -pts[1]['y'], -1, pts[1]['x']*pts[1]['yp'], pts[1]['y']*pts[1]['yp'], pts[1]['yp']],
         [-pts[2]['x'], -pts[2]['y'], -1, 0, 0, 0, pts[2]['x']*pts[2]['xp'], pts[2]['y']*pts[2]['xp'], pts[2]['xp']],
         [0, 0, 0, -pts[2]['x'], -pts[2]['y'], -1, pts[2]['x']*pts[2]['yp'], pts[2]['y']*pts[2]['yp'], pts[2]['yp']],
         [-pts[3]['x'], -pts[3]['y'], -1, 0, 0, 0, pts[3]['x']*pts[3]['xp'], pts[3]['y']*pts[3]['xp'], pts[3]['xp']],
         [0, 0, 0, -pts[3]['x'], -pts[3]['y'], -1, pts[3]['x']*pts[3]['yp'], pts[3]['y']*pts[3]['yp'], pts[3]['yp']],
         [-pts[4]['x'], -pts[4]['y'], -1, 0, 0, 0, pts[4]['x']*pts[4]['xp'], pts[4]['y']*pts[4]['xp'], pts[4]['xp']],
         [0, 0, 0, -pts[4]['x'], -pts[4]['y'], -1, pts[4]['x']*pts[4]['yp'], pts[4]['y']*pts[4]['yp'], pts[4]['yp']]]

    A = np.asarray(A)
    
    # Get eigen values of transpose(A)*A
    eigs_AT_A, vecs_AT_A = np.linalg.eig(A.T @ A)
    print(f"eigs_AT_A: {np.sqrt(np.abs(eigs_AT_A))}")

    # Sort eigen values in descending order and then further sort corresponding eigen vectors 
    sorted_indeces_AT_A = np.argsort(np.abs(eigs_AT_A))
    sorted_indeces_AT_A = sorted_indeces_AT_A[::-1]
    
    # Calculate singular values then generate sigma matrix 
    l_sig = np.sqrt(np.abs(eigs_AT_A))
    sig = l_sig[sorted_indeces_AT_A]*np.eye(len(eigs_AT_A))

    # Calculate U and V matrices
    V = np.asarray([vecs_AT_A[:,i] for i in sorted_indeces_AT_A]).T
    U = np.asarray([(1/l_sig[i])*A@vecs_AT_A[:,i] for i in sorted_indeces_AT_A]).T

    print("\n------------------------------------------------ Sig ------------------------------------------------")
    print(f"shape: {sig.shape}")
    print(pd.DataFrame(sig))

    print("\n------------------------------------------------ U ------------------------------------------------")
    print(f"shape: {U.shape}")
    print(pd.DataFrame(U))

    print("\n------------------------------------------------ V ------------------------------------------------")
    print(f"shape: {V.shape}")
    print(pd.DataFrame(V))

    print("\n------------------------------------------------ A ------------------------------------------------")
    A_df = pd.DataFrame(A)
    print(A_df)
    
    # Calculate then print estimated A
    print("\n------------------------------------------------ A_calc ------------------------------------------------")
    A_calc = U @ sig @ V.T
    A_calc_df = pd.DataFrame(A_calc)
    print(A_calc_df)

    # Homogenous Systems of Equations
    print("\n------------------------------------------------ Homography Matrix ------------------------------------------------")
    H = V[:, -1]
    H_mat = H.reshape(3,3)
    print(pd.DataFrame(H_mat))


    
