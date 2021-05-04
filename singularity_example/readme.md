# Guide for singularity example

Maybe explain here what the files are (too obvious?), how to build the container and how the container is run at test time?

Useful external resource: https://singularity-tutorial.github.io/

## Build
```
singularity build --fakeroot fets_simple_nnunet.sif container.def
```
With `--fakeroot`, you don't need to build with sudo (security option).
Tip for debugging building: use the `--sandbox` option

## Run
This is the command that will be executed at test time:
```
singularity run -c --writable-tmpfs --net --network=none --nv -B /path/to/test/data:/data:ro,/path/to/output/dir:/out_dir:rw fets_simple_nnunet.sif /data /out_dir
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
