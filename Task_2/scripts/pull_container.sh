# Can maybe even iterate through a file with urls and output filenames...
# Inputs:
# 1: project id
# 2: package name, version, file
# 3: output path
curl -K ~/fets_curl.cfg -o $3 "https://gitlab.hzdr.de/api/v4/projects/$1/packages/generic/$2"
