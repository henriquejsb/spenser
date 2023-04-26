import fast_denser
import sys 
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == '__main__':
	#folder = 'experiments'
	#os.system(f"rm -rf {folder}")
	fast_denser.engine.process_input(sys.argv[1:])
