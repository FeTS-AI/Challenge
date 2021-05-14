# Guide for singularity example

Maybe explain here what the files are, how to build the container and how the container is run at test time.

Useful external resource: https://singularity-tutorial.github.io/

## Build
```
singularity build --fakeroot container_simple.sif container_simple.def
```
With `--fakeroot`, you don't need to build with sudo (security option).
Tip for debugging building: use the `--sandbox` option

## Run
This is the command that will be executed at test time:
```
singularity run -c --writable-tmpfs --net --network=none --nv -B /path/to/test/data:/data:ro,/path/to/output/dir:/out_dir:rw container_simple.sif -i /data -o /out_dir
```
Description of the options:
- `--c` : TODO
- `--writable-tmpfs` : TODO
- `--net --network=none` : No network access
- `--nv` : use experimental GPU (nvidia) support
- `-B` : bind directory (src:dest:read-write-mode).

Tip for debugging runscript (from .def file): `singularity shell` instead of `singularity run` gives an interactive shell.

*TODO* Open questions:
- Maybe even `-C` (contain all) instead of `-c`? (recommended in tutorial) -> contain all works as well with my container
- in the tutorial, they also recommend `--no-home`. Necessary if `-c` or `-C` are set?
- Why do we need `--writable-tmpfs`? -> maybe temporary results can be saved (however, not much space available?)
-> ask David again and maybe kaapana guys about these options


# To do's
- The tmpfs when using writable-tmpfs seems to be very small. Not sure how to fix it, here are some ideas: 
  https://github.com/hpcng/singularity/issues/5718
  https://groups.google.com/a/lbl.gov/g/singularity/c/eq-tLo2SewM

  *update 21/04/29*: I'm not sure any more where this causes a problem. My container does not need the tmpfs.
- bootstrap image: Could there be problems with the nvidia driver? 
  From: nvcr.io/nvidia/pytorch:21.04-py3 worked on my workstation but I'm not sure if this was just because of "enhanced compatibility in the CUDA 11 versions.
  -> May have to specify minimum/maximum driver version as "HW requirement" for collaborators/participants
- team names: Teams  should add their team-identifier (letters only, no symbols) as a tag maybe? Or, depending on the submission platform, I will also assign them names (?)
- weird message upon container execution? (see https://forums.developer.nvidia.com/t/change-mofed-version-usage-example-message-with-tensorflow-ngc-container/124178)
