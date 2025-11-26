import numpy as np
try:
    from resample import jackknife
    print("resample.jackknife found")
    print(dir(jackknife))
    data = np.array([1, 2, 3, 4, 5])
    print("Testing jackknife.jackknife(np.mean, data)...")
    try:
        res = jackknife.jackknife(np.mean, data)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)
    
    print("Testing jackknife.bias(np.mean, data)...")
    try:
        res = jackknife.bias(np.mean, data)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)
        
    print("Testing jackknife.variance(np.mean, data)...")
    try:
        res = jackknife.variance(np.mean, data)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)
except ImportError:
    print("resample package not found")
