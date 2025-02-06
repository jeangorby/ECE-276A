import pickle
import sys
import time 

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset="6" # Uses only the first sequence of data
# cfile = "../data/cam/cam" + dataset + ".p"
ifile = "../data/imu/imuRaw" + dataset + ".p"
vfile = "../data/vicon/viconRot" + dataset + ".p"

ts = tic()
# camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

# print(imud.shape)
imuts = imud[0]
vicdts = vicd['ts'].transpose()
# print(imud[0][-1]-imud[0][0])
# print(vicdts[-1] - vicdts[0])

# print(vicd['rots'][:,:,1171])
# print(imud[0][-1]-imud[0][0])
import numpy as np
# np.savetxt('imud2.txt', np.transpose(imud))

# print(vicd['rots'].shape[-1])
# print(np.transpose(vicd['rots'])[0])
# np.savetxt('vicdrots5.txt', np.transpose(vicd['rots']))

# Write the array to disk
with open('vicdrots6.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(vicd['rots'].shape))
    
    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for i in range(vicd['rots'].shape[-1]):

        data_slice = vicd['rots'][:,:,i]
        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%10.8f')

        # Writing out a break to indicate different slices...
        outfile.write('index = {0}\n'.format(i+1))

# print(camd['cam'].shape)
# print(camd['ts'][0])
# np.savetxt('camdts.txt', np.transpose(camd['ts']), fmt='%17.6f')