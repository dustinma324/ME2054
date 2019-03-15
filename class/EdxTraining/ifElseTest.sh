#!/bin/bash

echo Give your name

read name

if [ "$name" == Dustin ] ; then
	echo Hello $name
elif [ "$name" == Tim ]; then
	echo Hello $name
else
	echo Forget it $name, you are not important
fi
