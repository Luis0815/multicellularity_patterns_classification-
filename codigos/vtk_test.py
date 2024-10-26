import matplotlib.pyplot as plt
import numpy as np

import os
import re

filename = "-4_0_-8.vtk"

with open(filename, "r") as f:
  lines = f.readlines()

  # make auxiliar variables
  flag_section = ""
  point_types = []
  point_ids = []
  nx, ny = 0, 0
  
  #iterate over the lines
  for line in lines:
    # if line contains numbers pass
    if re.match(r"^[\d\s]+$", line):
      # remove the line break and split the line
      line = line.strip()
      if flag_section == "CellType":
        # append the cell types
        point_types += [int(i) for i in line.split()]
      elif flag_section == "CellId":
        # append the cell ids
        point_ids += [int(i) for i in line.split()]
      pass
    # else update the flag
    else:
      flag_section = line.split(" ")[0]
      if flag_section == "DIMENSIONS":
        # get the number of cells
        nx, ny = line.split(" ")[1:3]
      pass
  pass

ids = np.array(point_ids).reshape(int(nx), int(ny))
# plot the cells where id == x for x in the first 5 ids
_ids = list(set(point_ids))
m = [ids == x for x in _ids[:100]]
m = [mm*(i+1) for i,mm in enumerate(m)]
plt.imshow(sum(m))

types = np.array(point_types).reshape(int(nx), int(ny))
# plot all the cells
plt.imshow(types)

pass
