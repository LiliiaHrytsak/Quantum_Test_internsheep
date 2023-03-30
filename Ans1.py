import time

start_time = time.time()

def fsd(n):
    if (n < 10000000) and (n > 0):
        return n * (n + 1) / 2
    else:
        return ('Invalid input')


print(fsd(10000000000))
end_time = time.time()
print("Total execution time: ", end_time - start_time)