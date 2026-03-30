KEEP=89240
scancel $(squeue -h -u "$USER" -o "%A" | awk -v k="$KEEP" '$1!=k')