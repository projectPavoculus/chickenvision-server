import sys

# Prints a progress bar to the console.
def loadbar(current, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filledLength = int(length * current // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if current == total:
        print()


# Counts the number of lines in a file.
def count_lines(file):
    return sum(1 for line in file)


# Calculates the size of an object in kilobytes or megabytes.
def object_size(obj):
    if sys.getsizeof(obj) < 1024:
        return f'{(sys.getsizeof(obj) / (1024.0)):.2f} KB'
    else:
        return f'{(sys.getsizeof(obj) / (1024.0 * 1024.0)):.2f} MB'
