import sys
import subprocess
import time

def main():
    # Ask the user for their operating system
    print("Select your operating system:")
    print("1: Windows")
    print("2: Linux")
    os_choice = input("Enter your choice (1/2): ").strip()

    # Decide which Python command to use based on the operating system choice
    python_command = 'python3' if os_choice == '2' else 'python'

    # Ask the user for the type of testing
    print("Select the type of testing:")
    print("1: Basic Testing")
    print("2: Advanced Testing")
    test_choice = input("Enter your choice (1/2): ").strip()

    # Decide which script to run based on the user's choice
    if test_choice == '1':
        print("Running testing script for BASIC testing")
        time.sleep(3)
        subprocess.run([python_command, 'src/t1.py'])
    elif test_choice == '2':
        print("Running testing script for ADVANCED testing")
        time.sleep(3)
        subprocess.run([python_command, 'src/t2.py'])
    else:
        print("Invalid testing choice. Please use 1 or 2.")
        sys.exit(1)

if __name__ == "__main__":
    main()
