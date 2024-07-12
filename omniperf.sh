#!/bin/bash

set -eux

executable="$EXECUTABLE"
profile_name="$PROFILE_NAME"


omniperf profile \
  -n "$profile_name" -- "$executable"

profile_path="workloads/$profile_name"

some_profile_file="$(find "$profile_path" -name pmc_perf.csv)"
profile_files_path="$(basename "$some_profile_file")"
analysis_path="${profile_path}/analysis.txt"

omniperf analyze -q -p "${profiles_files_path}" > "$analysis_path"
