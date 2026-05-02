"""
color_logs.py
Description: Controller for Color within the terminal.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

# Red
def print_error(s):
    print("\033[91m {}\033[00m".format(s))

# Green
def print_succ(s):
    print("\033[92m {}\033[00m".format(s))

# Yellow (The best 'Orange-ish' high-visibility choice)
def print_warning(s):
    print("\033[93m {}\033[00m".format(s))

# Blue
def print_blue(s):
    print("\033[94m {}\033[00m".format(s))