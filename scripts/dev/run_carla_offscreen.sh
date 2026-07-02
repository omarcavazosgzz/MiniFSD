#!/bin/bash

cd ~/Simulators

./CarlaUE4.sh \
  -RenderOffScreen \
  -quality-level=Low \
  -benchmark \
  -fps=20 \
  -nosound
