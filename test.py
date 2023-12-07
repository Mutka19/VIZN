import sys
import subprocess
import time

def main():
    # Check if the user has provided an argument
    if len(sys.argv) != 2:
        print("Usage: python launcher.py '1' OR '2'")
        print("Use 1 for basic testing")
        print("Use 2 for advanced testing")
        print("Same training and datasets, it's just the testing functionlity/methodology is different")
        sys.exit(1)

    # Get the argument passed by the user
    arg = sys.argv[1]

    # Decide which script to run based on the argument
    if arg == '1':
        print("Running testing script for BASIC testing")
        time.sleep(3)
        subprocess.run(['python', 'src/t1.py'])
    elif arg == '2':
        print("Running testing script for ADVANCED testing")
        time.sleep(3)
        subprocess.run(['python', 'src/t2.py'])
    else:
        print("Invalid argument. Please use 1 or 2.")
        sys.exit(1)

if __name__ == "__main__":
    main()
