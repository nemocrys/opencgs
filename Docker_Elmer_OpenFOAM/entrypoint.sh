#!/bin/bash

USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting with UID: $USER_ID, GID: $GROUP_ID"
useradd -u $USER_ID -o -m user
groupmod -g $GROUP_ID user
export HOME=/home/user

# quality of life improvements
echo 'bind '"'"'"\e[A": history-search-backward'"'" >> $HOME/.bashrc 
echo 'bind '"'"'"\e[B": history-search-forward'"'" >> $HOME/.bashrc 
echo 'LS_COLORS=$LS_COLORS:'"'"'di=32:'"'"' ; export LS_COLORS' >> $HOME/.bashrc 
echo "alias of21='source /usr/lib/openfoam/openfoam2112/etc/bashrc '" >> $HOME/.bashrc

# OpenFOAM enviroment variables
export WM_USER_PROJECT_DIR=/home/workdir/user-foam
export FOAM_RUN=/home/workdir/run

exec /usr/sbin/gosu user "$@"
