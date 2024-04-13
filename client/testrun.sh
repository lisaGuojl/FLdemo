#!/bin/bash

NUM=50

trap ctrl_c INT

function ctrl_c() {
  pkill -P $$
}

for ((i=0; i<NUM; i++)); do
  sleep 3
  python3 app/docker-client.py 
done
wait
exit
