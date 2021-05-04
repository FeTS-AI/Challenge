What do I need?

Singularity example:
- [] See https://phabricator.mitk.org/source/mood/browse/master/docker_example/ for reference
- [] Interface definition: How is the container run at test time? -> singularity run .... ?
- [] Example definition file
- [] Example prediction script
- [] Guide how to build the singularity container (website?)
- [] Common issues (website)

Scripts:
- test_container: similar to https://phabricator.mitk.org/source/mood/browse/master/scripts/test_docker.py -> would be best if we also integrated our evaluation code into this!

Internal functions for testing submissions:
- guide how to pull submitted containers
- script that executes test runs on the cluster. Very similar to test_container!