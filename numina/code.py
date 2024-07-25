
try:
    from sympy import *

    from math import factorial
    
    def sum_of_digits(n):
        return sum(int(digit) for digit in str(n))
    
    # Test the functions
    sparkle_operation = lambda n: factorial(sum_of_digits(n))
    
    # Initial test cases
    assert sparkle_operation(13) == factorial(4) == 24
    assert sparkle_operation(98) == factorial(17) == 3556872
    
    print("Sparkle operation works correctly.")
    
except Exception as e:
    print(e)
    print('FAIL')
