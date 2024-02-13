#!/bin/bash

exit_requested=0

while [ $exit_requested -eq 0 ]
do
    venv/bin/python main.py

    if test -f "exit-requested.txt"; then
        exit_requested=1
    fi
done

rm exit-requested.txt
