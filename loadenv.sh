#!/bin/bash

# Load all .env files from the top level of the project
for file in *.env; do
  export $(echo "$file" | sed 's/.env$//')_$(cat "$file" | sed 's/^/export /')
done

