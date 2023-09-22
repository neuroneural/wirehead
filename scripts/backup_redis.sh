#!/bin/bash
# Copy data from one Redis database to another

source_db=0
target_db=1

# Fetch all keys from source db
keys=$(redis-cli -n $source_db KEYS '*')

# Copy each key-value pair to target db
for key in $keys; do
    val=$(redis-cli -n $source_db DUMP "$key")
    ttl=$(redis-cli -n $source_db TTL "$key")
    if [ $ttl -lt 0 ]; then
	redis-cli -n $target_db RESTORE "$key" 0 "$val"
    else
	redis-cli -n $target_db RESTORE "$key" $(( ttl * 1000 )) "$val"
    fi
done

