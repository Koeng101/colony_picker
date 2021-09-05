"""A set of helper functions for running long tests.
"""
import sys
from tests import PRINT_PROGRESS


def draw_loading_bar(iter: int, total_iters: int, func_name: str):
    """Meant to be called every iteration of a loop to display the progress of a
    test as an animated loading bar in the terminal. 

    Args:
        iter (int): The iteration of the for loop *indexed from zero*.
        total_iters (int): The total number of iterations of the for loop *not
            indexed from zero*.
        func_name (str): The name of the test function you are drawing the
            loading bar for.
    """
    if PRINT_PROGRESS:
        loading_bar_size = len(func_name) - 1
        completed_iters = int(((iter + 1)/total_iters)*loading_bar_size)
        completed_iters_str = "="*completed_iters
        remaining_iters_str = " "*(loading_bar_size - completed_iters)

        sys.stdout.write(
            f"\r[{completed_iters_str}>{remaining_iters_str}]")
        sys.stdout.flush()


def print_decorated_title(func_name: str):
    """Meant to print the title of the test that is running in the terminal with
    a pretty border. Often coupled with the loading bar animation to display
    the test that the loading bar is associated with. 

    Args:
        func_name (str): The name of the test you are displaying the title of.
    """
    if PRINT_PROGRESS:
        title_len = len(func_name) + 2
        border_str = "*"*title_len
        print()
        print()
        print(border_str)
        print(f"~{func_name}~")
        print(border_str)
