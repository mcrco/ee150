import os
from glob import glob
import subprocess
import json
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)
result_template = {"tests": []}
global_results = {
    "test_2_prelim": {
        "number": "2.prelim",
        "score": 0.0,
        "max_score": 0.0,
        "status": "failed",
    },
    "test_2a": {
        "number": "2.a",
        "score": 0.0,
        "max_score": 2.0,
    },
    "test_2b": {
        "number": "2.b",
        "score": 0.0,
        "max_score": 2.0,
    },
    "test_2c": {
        "number": "2.c",
        "score": 0.0,
        "max_score": 2.0,
    },
    "test_2d": {
        "number": "2.d",
        "score": 0.0,
        "max_score": 2.0,
    },
    "test_2e": {
        "number": "2.e",
        "score": 0.0,
        "max_score": 2.0,
    },
    "test_2f": {
        "number": "2.f",
        "score": 0.0,
        "max_score": 2.0,
    },
    "test_2g": {
        "number": "2.g",
        "score": 0.0,
        "max_score": 2.0,
    },
    "test_2h": {
        "number": "2.h",
        "score": 0.0,
        "max_score": 3.0,
    },
    "test_2i": {"number": "2.i", "score": 0.0, "max_score": 0.0, "status": "failed"},
    "test_3a": {
        "number": "3.a",
        "score": 0.0,
        "max_score": 5.0,
    },
    "test_3b": {
        "number": "3.b",
        "score": 0.0,
        "max_score": 5.0,
    },
    "test_3c": {
        "number": "3.c",
        "score": 0.0,
        "max_score": 5.0,
    },
}

# Check if the AUTOGRADER environment variable is set to a truthy value
AUTOGRADER_MODE = bool(os.getenv("AUTOGRADER", "").lower() in ["1", "true", "yes"])


def colorize(text, color):
    """
    Returns colorized text unless AUTOGRADER_MODE is enabled (not used)
    """
    return text if AUTOGRADER_MODE else f"{color}{text}{Style.RESET_ALL}"


def main():
    tests = sorted(glob("grad/tests/*.py"))
    for test_fpath in tests:
        print(colorize(f"Running test: {test_fpath}", Fore.LIGHTBLUE_EX))
        try:
            res = process_test(test_fpath)
            print_test_results(res)
        except Exception as e:
            print(
                colorize(
                    e,
                    Fore.YELLOW,
                )
            )

    # Write global results to JSON file
    with open("results.json", "w") as f:
        json.dump(result_template, f, indent=4)
        print()
        print(Fore.LIGHTMAGENTA_EX + "Results written to results.json")

    # Cleanup
    if os.path.exists(".report.json"):
        os.remove(".report.json")


def process_test(fpath):
    """
    Runs pytest on the given file and captures the JSON report.
    """
    result = subprocess.run(
        ["pytest", "--json-report", "-q", fpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # captures any bad return codes
    if result.returncode in [2, 3, 4, 5]:
        raise Exception(
            f"PANIC! pytest returned with exit code {result.returncode} see https://docs.pytest.org/en/stable/reference/exit-codes.html"
        )
    # Load the results from the default `.report.json`
    with open(".report.json", "r") as f:
        report = json.load(f)

    test_id = os.path.splitext(os.path.basename(fpath))[0]
    passed, count = 0, 0
    for test in report.get("tests", []):
        count += 1
        if test["outcome"] == "passed":
            passed += 1

    if passed == count:
        # all tests passed
        global_results[test_id]["score"] = global_results[test_id]["max_score"]
        global_results[test_id]["status"] = "passed"
    result_template["tests"].append(global_results[test_id])

    return report


def print_test_results(report):
    """
    Prints test results with colored output.
    """
    if "error" in report:
        print(colorize(f"Error: {report['error']}", Fore.RED))
        return

    for test in report.get("tests", []):
        try:
            test_id = test["nodeid"]
            outcome = test["outcome"]
            if outcome == "passed":
                print(colorize(f"[PASS] {test_id}", Fore.GREEN))
            elif outcome == "failed":
                print(colorize(f"[FAIL] {test_id}", Fore.RED))
            else:
                print(colorize(f"[{outcome.upper()}] {test_id}", Fore.YELLOW))
        except:
            print(
                colorize(
                    "PANIC! -> something failed, are you sure you followed the setup correctly?",
                    Fore.YELLOW,
                )
            )


if __name__ == "__main__":
    main()
