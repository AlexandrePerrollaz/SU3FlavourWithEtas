import numpy as np
from iminuit import Minuit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy import stats
from chi2_functions import * # Assuming chi2 is defined to accept 25 parameters

number_guess = 5000
num_param = 13
dof = 36

def minimize_with_guess(guess):
    try:
        # Initialize Minuit with parameters from guess
        minuit = Minuit(
            chi2MinuitNonet,
            ampT9X9=guess[0], ampC9X9=guess[1], ampPuc9X9=guess[2], ampA9X9=guess[3],
            ampPAuc9X9=guess[4], ampPtc9X9=guess[5], ampPAtc9X9=guess[6],
            delC9X9=guess[7], delPuc9X9=guess[8], delA9X9=guess[9],
            delPAuc9X9=guess[10], delPtc9X9=guess[11], delPAtc9X9=guess[12]
        )
        # Perform minimization
        minuit.limits["ampT9X9"] = (0,None)
        minuit.limits["ampC9X9"] = (0,None)
        minuit.limits["ampPuc9X9"] = (0,None)
        minuit.limits["ampA9X9"] = (0,None)
        minuit.limits["ampPAuc9X9"] = (0,None)
        minuit.limits["ampPtc9X9"] = (0,None)
        minuit.limits["ampPAtc9X9"] = (0,None)

        minuit.limits["delC9X9"] = (0,2*np.pi)
        minuit.limits["delPuc9X9"] = (0,2*np.pi)
        minuit.limits["delA9X9"] = (0,2*np.pi)
        minuit.limits["delPAuc9X9"] = (0,2*np.pi)
        minuit.limits["delPtc9X9"] = (0,2*np.pi)
        minuit.limits["delPAtc9X9"] = (0,2*np.pi)

        minuit.migrad()

        # Return successful result
        return {
            'guess': guess,
            'minimum': [minuit.values[k] for k in minuit.parameters],
            'chi2': minuit.fval,
        }

    except Exception as e:
        print(f"Error during minimization for guess {guess[:5]}: {e}")
        return {'guess': guess, 'error': str(e)}

if __name__ == "__main__":
    try:
        # Generate random initial guesses
        initial_guesses = 2 * np.pi * np.random.rand(number_guess, num_param)

        # Parallel execution
        results = []
        with ProcessPoolExecutor() as executor:
            for res in tqdm(executor.map(minimize_with_guess, initial_guesses), total=len(initial_guesses)):
                if res:  # Ensure the result is not None
                    results.append(res)

        # Filter out failed minimizations
        successful_results = [res for res in results if 'error' not in res]

        # Find the result with the lowest chi2
        if successful_results:
            best_result = min(successful_results, key=lambda r: r['chi2'])
            print("\nBest Result:")
            print(f"Guess: {best_result['guess']}, Minimum: {best_result['minimum']}, Chi2: {best_result['chi2']}")
            print("p-value:", stats.chi2.sf(best_result['chi2'],df = dof))
        else:
            print("No successful minimizations.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")

