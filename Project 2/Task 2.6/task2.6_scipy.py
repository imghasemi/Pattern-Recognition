from scipy import spatial
import numpy as np
from matplotlib import pyplot as plt
import time

# train and test file str
train_f = "data2-train.dat"
test_f  = "data2-test.dat"
save = True

# reads first and second column data and labels from the given structured file
def read_file(file_name): 
    # read data as 3D array of data type 'object'
    data = np.loadtxt(file_name, dtype=np.object, comments='#', delimiter=None)
        
    # read first and second column data and labels into 3D array 
    data = data.astype(np.float)
    data[:,-1] = data[:,-1].astype(np.int32) 
    
    return data

# generate kd-tree
train_data = read_file(train_f)
tree = spatial.KDTree(train_data[:,[0,1]])
test_data = read_file(test_f)

# benchmarking array
bench_m = np.zeros(len(test_data))

for i in range(1,len(test_data)):
    start = time.time()
    s = test_data[i]
    # query kd_tree
    d, NN = tree.query([s[0],s[1]])

    done = time.time()
    bench_m[i] = bench_m[i-1] + done - start

if save:
    # plot overall run time for computing
    fig = plt.figure()
    axs = fig.add_subplot(111)
    x = np.linspace(0, len(bench_m), len(test_data))
    axs.plot(x, bench_m*1000, label='benchmark(msec)')
    plt.xlabel('samples')
    plt.ylabel('msec')
    axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    filename = 'chart/benchmark_scipy.pdf'
    plt.savefig(filename, facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
