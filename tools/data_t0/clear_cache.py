from data_utils import DATA_CONFIG_T0
import sys
import os

if sys.argv[1] == "all":
    for name in DATA_CONFIG_T0:
        data_dir = DATA_CONFIG_T0[name]["data_dir"]
        os.system("rm {}".format(os.path.join(data_dir, "*.pkl")))
else:
    for name in sys.argv[4:]:
        data_dir = DATA_CONFIG_T0[name]["data_dir"]
        print(name)
        os.system("rm {}".format(os.path.join(data_dir, "cache_train_1.0_{}.pkl".format(sys.argv[1]))))
        os.system("rm {}".format(os.path.join(data_dir, "cache_valid_1.0_{}.pkl".format(sys.argv[2]))))
        os.system("rm {}".format(os.path.join(data_dir, "cache_test_1.0_{}.pkl".format(sys.argv[3]))))
