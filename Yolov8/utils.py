def loadbar(current, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """Prints a progress bar to the console."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filledLength = int(length * current // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if current == total:
        print()

def count_lines(file):
    """Counts the number of lines in a file."""
    return sum(1 for line in file)