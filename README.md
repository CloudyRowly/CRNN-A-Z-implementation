# CRNN-A-Z-implementation

Implemented multiprocessing for convolutioning
Testing on a 5335x3557 image:
single core: 154.17s
8 cores    : 21.14 s
~ 7.3 times faster for large images

Small images may convolute faster on single core as divde and allocate jobs to cores take time. Current minimum package size for a core is 256 pixels vertically. 
