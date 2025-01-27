import numpy as np
from iminuit import Minuit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy import stats
from chi2_functions import chi2,chi2Minuit,chi2MinuitWithoutEta,chi2MinuitNoEtaEta,chi2MinuitNoEtaEtaDS1only,chi2MinuitNoEtaEtaToverCfixed  # Assuming chi2 is defined to accept 25 parameters

number_guess = 2000
num_param = 19
dof = 25

def minimize_with_guess(guess):
    try:
        # Initialize Minuit with parameters from guess
        minuit = Minuit(
            chi2MinuitNoEtaEtaToverCfixed,
            ampT8X8=guess[0], ampPuc8X8=guess[1], ampA8X8=guess[2],
            ampPAuc8X8=guess[3], ampPtc8X8=guess[4], ampPAtc8X8=guess[5],
            delC8X8=guess[6], delPuc8X8=guess[7], delA8X8=guess[8],
            delPAuc8X8=guess[9], delPtc8X8=guess[10], delPAtc8X8=guess[11],
            ampT8X1=guess[12], ampPuc8X1=guess[13],
            ampPtc8X1=guess[14], delT8X1=guess[15], delC8X1=guess[16],
            delPuc8X1=guess[17], delPtc8X1=guess[18]
        )
        # Perform minimization
        minuit.limits["ampT8X8"] = (0,None)
        minuit.limits["ampPuc8X8"] = (0,None)
        minuit.limits["ampA8X8"] = (0,None)
        minuit.limits["ampPAuc8X8"] = (0,None)
        minuit.limits["ampPtc8X8"] = (0,None)
        minuit.limits["ampPAtc8X8"] = (0,None)

        minuit.limits["ampT8X1"] = (0,None)
        minuit.limits["ampPuc8X1"] = (0,None)
        minuit.limits["ampPtc8X1"] = (0,None)

        minuit.limits["delC8X8"] = (0,2*np.pi)
        minuit.limits["delPuc8X8"] = (0,2*np.pi)
        minuit.limits["delA8X8"] = (0,2*np.pi)
        minuit.limits["delPAuc8X8"] = (0,2*np.pi)
        minuit.limits["delPtc8X8"] = (0,2*np.pi)
        minuit.limits["delPAtc8X8"] = (0,2*np.pi)

        minuit.limits["delT8X1"] = (0,2*np.pi)
        minuit.limits["delC8X1"] = (0,2*np.pi)
        minuit.limits["delPuc8X1"] = (0,2*np.pi)
        minuit.limits["delPtc8X1"] = (0,2*np.pi)
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

