
# Copyright 2019 Filipe Assuncao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
import random
from time import time
import numpy as np
import os

import gc
import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from typing import OrderedDict
from torch.utils.data import DataLoader
#import tonic
from snntorch import spikegen


import contextlib
from time import time as t
#from tqdm import tqdm


DEBUG = True

#torch.set_num_threads(os.cpu_count() - 1)

