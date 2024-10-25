import numpy as np
import matplotlib.pyplot as plt


class Solutions:

    total_steps = 100
    num_points = 1000

    def __init__(self) -> None:
        print('''
              The solutions are presented as follows:
              ''')

    @staticmethod
    def case(number):
        if number == 1:
            a, b, c = 1, 1, 1
        elif number == 2:
            a, b, c = 1, -1, 1
        elif number == 3:
            a, b, c = 1, 1, -1
        return a, b, c

    @staticmethod
    def f1(x, case_number=1):
        a, b, c = Solutions.case(case_number)
        return a*np.abs(x - b) + c

    @staticmethod
    def grad1(x, case_number=1):
        a, b, c = Solutions.case(case_number)
        if x-b >= 0:
            return a
        return -a

    def exercise1(self):
        num_cases = 3

        x = np.linspace(-2, 2, Solutions.num_points)
        y_s = [Solutions.f1(x, case_number=i+1) for i in range(num_cases)]

        x_run = 1
        values_history = {i+1: np.zeros(Solutions.total_steps) for i in range(num_cases)}
        errors_history = {i+1: np.zeros(Solutions.total_steps) for i in range(num_cases)}

        for case in range(1, num_cases+1):
            for i in range(Solutions.total_steps):
                a, b, c = Solutions.case(case)
                x_star = b
                values_history[case][i] = x_run
                errors_history[case][i] = np.abs(x_star - x_run)

                x_run -= (1/(2*np.sqrt(i+1)))*Solutions.grad1(x_run, case)

            print('The Value of x is: {0:.3f} The value reached is {1:.3f}'.format(x_star, x_run))

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for i in range(num_cases):
            axs[0, i].plot(Solutions.f1(values_history[i+1], i+1))
            axs[0, i].set(xlabel='x value', ylabel='f(x)')
            axs[0, i].set_title('case {}'.format(i+1))
            axs[0, i].grid()

            axs[1, i].plot(errors_history[i+1])
            axs[1, i].set(xlabel='step', ylabel='error')
            axs[1, i].grid()
        plt.show()

    @staticmethod
    def f2(x, case_number):
        a, b, c = Solutions.case(case_number)
        return a*(x**2)+b*x+c

    @staticmethod
    def grad2(x, case_number=1):
        a, b, c = Solutions.case(case_number)
        return 2*a*x+b

    def exercise2(self):
        num_cases = 3

        for case in range(1, num_cases+1):
            x = np.linspace(-2, 2, Solutions.num_points)
            y_s = [Solutions.f2(x, case_number=i+1) for i in range(num_cases)]

            values_history = {i+1: np.zeros(Solutions.total_steps) for i in range(num_cases)}
            errors_history = {i+1: np.zeros(Solutions.total_steps) for i in range(num_cases)}
            alphas = [0.05, 0.95, 1.05]

            for alpha_case in range(1, num_cases+1):
                x_run = 1
                alpha = alphas[alpha_case - 1]
                for i in range(Solutions.total_steps):
                    a, b, c = Solutions.case(case)
                    x_star = -b/(2*a)
                    values_history[alpha_case][i] = x_run
                    errors_history[alpha_case][i] = np.abs(x_star - x_run)

                    x_run -= alpha*Solutions.grad2(x_run, case)

                print('The Value of x is: {0:.3f}\t f(x)={1:.3f}\t The value reached is {2:.3f}\t of case {3}'.format(
                    x_star, Solutions.f2(x_star, case), Solutions.f2(x_star, case), x_run, alpha_case))

            fig, axs = plt.subplots(2, 3, figsize=(15, 8))
            for alpha_case in range(num_cases):
                axs[0, alpha_case].plot(Solutions.f2(values_history[alpha_case+1], alpha_case+1))
                axs[0, alpha_case].set(xlabel='x value', ylabel='f(x)')
                axs[0, alpha_case].set_title('case {} alpha {}'.format(case, alphas[alpha_case]))
                axs[0, alpha_case].grid()

                axs[1, alpha_case].plot(errors_history[alpha_case+1])
                axs[1, alpha_case].set(xlabel='step', ylabel='error')
                axs[1, alpha_case].grid()
            plt.show()

    @staticmethod
    def f3(x):
        return x**4 + 0.5*(x**3) - 4*(x**2)

    @staticmethod
    def grad3(x):
        return 4*(x**3)+1.5*(x**2)-8*x

    def exercise3(self):
        num_cases = 3
        alpha = 1/(2*4) - .1
        betas = [0.1, 0.9, 1.05]

        x_loc_min = 1.239
        x_glb_min = x_star = -1.614

        x = np.linspace(-2, 2, Solutions.num_points)
        y_s = Solutions.f3(x)

        values_history = {betas[i]: np.array([]) for i in range(num_cases)}
        errors_history = {betas[i]: np.zeros(Solutions.total_steps) for i in range(num_cases)}

        for beta_case in range(num_cases):
            x_run = x_prev = 2
            values_history[betas[beta_case]] = np.append(values_history[betas[beta_case]], x_run)

            beta = betas[beta_case]
            for i in range(Solutions.total_steps):
                errors_history[betas[beta_case]][i] = np.abs(x_star - x_run)
                x_run -= alpha*Solutions.grad3(x_run) - beta*(x_run - x_prev)
                values_history[betas[beta_case]] = np.append(values_history[betas[beta_case]], x_run)
                x_prev = values_history[betas[beta_case]][-2]

            print('The Value of x is: {0:.3f} The value reached is {1:.3f}, Error: {2:.3f}'.format(
                x_star, x_run, np.abs(x_star - x_run)))

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for beta_case in range(num_cases):
            axs[0, beta_case].plot(Solutions.f3(values_history[betas[beta_case]]))
            axs[0, beta_case].set(xlabel='x value', ylabel='f(x)')
            axs[0, beta_case].set_title('case {} beta {}'.format(beta_case, betas[beta_case]))
            axs[0, beta_case].grid()

            axs[1, beta_case].plot(errors_history[betas[beta_case]])
            axs[1, beta_case].set(xlabel='step', ylabel='error')
            axs[1, beta_case].grid()
        plt.show()

    @staticmethod
    def f4(x):
        return (x**4)+0.5*(x**3)-4*(x**2)

    @staticmethod
    def grad4(x):
        return 4*(x**3)+1.5*(x**2)-8*x

    @staticmethod
    def hessian4(x):
        return 12*(x**2)+3*(x)-8

    def exercise4(self):
        num_cases = 2

        x = np.linspace(-2, 2, Solutions.num_points)
        y_s = Solutions.f4(x)

        x_loc_min = 1.239
        x_glb_min = x_star = -1.614

        x_runs = [2, -3]
        values_history = {x_runs[i]: np.zeros(Solutions.total_steps) for i in range(num_cases)}
        errors_history = {x_runs[i]: np.zeros(Solutions.total_steps) for i in range(num_cases)}

        for case, x_run in enumerate(x_runs):
            for i in range(Solutions.total_steps):
                values_history[x_runs[case]][i] = x_run
                errors_history[x_runs[case]][i] = np.abs(x_star - x_run)

                x_run = x_run - Solutions.grad4(x_run)/Solutions.hessian4(x_run)

            print('The Value of x is: {0:.3f} The value reached is {1:.3f}'.format(x_star, x_run))

        fig, axs = plt.subplots(3, 2, figsize=(11, 15))
        for case in range(num_cases):
            axs[0, case].plot(Solutions.f4(values_history[x_runs[case]]))
            axs[0, case].set(xlabel='x value', ylabel='f(x)')
            axs[0, case].set_title('case {} start {}'.format(case, x_runs[case]))
            axs[0, case].grid()

            axs[1, case].plot(errors_history[x_runs[case]])
            axs[1, case].set(xlabel='step', ylabel='error')
            axs[1, case].grid()

            axs[2, case].plot(errors_history[x_runs[case]][:-1]/(errors_history[x_runs[case]][1:]**2))
            axs[2, case].set(xlabel='step', ylabel=r"$\frac{e_{t}}{e_{t+1}^2}$")
            axs[2, case].grid()

        plt.show()


if __name__ == '__main__':
    answers = Solutions()
    answers.exercise1()
    answers.exercise2()
    answers.exercise3()
    answers.exercise4()
