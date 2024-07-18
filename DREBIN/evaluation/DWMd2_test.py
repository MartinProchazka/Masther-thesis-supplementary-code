from DWM_main import main
from multiprocessing import Pool
import random
random.seed(42)

def threads():
    random_seeds = [random.randint(0, 10000) for _ in range(10)]
    args = [[seed, 1, 0] for seed in random_seeds]
    results = []
    with Pool() as pool:
        results = pool.map(main, args)
    print(results)



if __name__ == "__main__":
    threads()