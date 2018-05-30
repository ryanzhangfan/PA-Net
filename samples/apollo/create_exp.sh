#!/bin/bash

set -u

if [ ! -d "exp/" ]; then
  mkdir exp
fi

if [ ! -d "submit/" ]; then
  mkdir submit
fi

mkdir "exp/"$1
cp template/*.py "exp/"$1
cp template/*.sh "exp/"$1
