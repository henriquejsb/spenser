import fast_denser
import sys 
import os

if __name__ == '__main__':
	#folder = 'experiments'
	#os.system(f"rm -rf {folder}")
	fast_denser.engine.process_input(sys.argv[1:])