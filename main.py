from neurons import SimpleAI
import numpy as np


def main():
    np.random.seed(121)

    sai = SimpleAI()

    inpt_data = np.array([[0.58, 3.38, 0.91], [3.38, 0.91, 5.80], [0.91, 5.80, 0.91], [5.80, 0.91, 5.01],
                          [0.91, 5.01, 1.17], [5.01, 1.17, 4.67], [1.17, 4.67, 0.60], [4.67, 0.60, 4.81],
                          [0.60, 4.81, 0.53], [4.81, 0.53, 4.75]])

    reslt_data = np.array([[5.80], [0.91], [5.01], [1.17], [4.67], [0.60], [4.81], [0.53], [4.75], [1.01]])

    inpt_data_check = np.array([[0.53, 4.75, 1.01], [4.75, 1.01, 5.04]])
    reslt_data_check = np.array([[5.04], [1.07]])


    sai.study(input_data=inpt_data, output_data=reslt_data)
    print("\nExpect: " + str(np.concatenate(reslt_data_check, axis=None)))
    print("\nCurrent: " + str(np.concatenate(sai.forward_propagation(inpt_data_check)["out"], axis=None)))
    print("\nMistake: " + str(np.concatenate((sai.forward_propagation(inpt_data_check)["out"] - reslt_data_check)**2, axis=None)))


if __name__ == '__main__':
    main()
