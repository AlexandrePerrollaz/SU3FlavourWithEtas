import numpy as np
from iminuit import Minuit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from chi2_functions import chi2,chi2Minuit,chi2MinuitWithoutEta,chi2MinuitNoEtaEta  # Assuming chi2 is defined to accept 25 parameters

def minimize_with_guess(guess):
    try:
        # Initialize Minuit with parameters from guess
        minuit = Minuit(
            chi2MinuitNoEtaEta,
            ampT8X8=guess[0], ampC8X8=guess[1], ampPuc8X8=guess[2], ampA8X8=guess[3],
            ampPAuc8X8=guess[4], ampPtc8X8=guess[5], ampPAtc8X8=guess[6],
            delC8X8=guess[7], delPuc8X8=guess[8], delA8X8=guess[9],
            delPAuc8X8=guess[10], delPtc8X8=guess[11], delPAtc8X8=guess[12],
            ampT8X1=guess[13], ampC8X1=guess[14], ampPuc8X1=guess[15],
            ampPtc8X1=guess[16], delT8X1=guess[17], delC8X1=guess[18],
            delPuc8X1=guess[19], delPtc8X1=guess[20]
        )
        # Perform minimization
        minuit.limits["ampT8X8"] = (0,None)
        minuit.limits["ampC8X8"] = (0,None)
        minuit.limits["ampPuc8X8"] = (0,None)
        minuit.limits["ampA8X8"] = (0,None)
        minuit.limits["ampPAuc8X8"] = (0,None)
        minuit.limits["ampPtc8X8"] = (0,None)
        minuit.limits["ampPAtc8X8"] = (0,None)

        minuit.limits["ampT8X1"] = (0,None)
        minuit.limits["ampC8X1"] = (0,None)
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
        initial_guesses = 2 * np.pi * np.random.rand(500, 25)

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
        else:
            print("No successful minimizations.")

    except Exception as e:
        print(f"An error occurred during execution: {e}")

