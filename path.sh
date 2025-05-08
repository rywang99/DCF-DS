#
export KALDI_ROOT=

export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH


export PATH=/opt/lib/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/lib/cuda-10.2/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PATH=/home4/intern/stniu/anaconda3/envs/mamse/bin/:$PATH
export LD_LIBRARY_PATH=/home4/intern/stniu/anaconda3/envs/mamse/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home4/intern/mkhe/anaconda3/lib:$LD_LIBRARY_PATH

export PATH=/home4/intern/stniu/libs/ffmpeg/bin/:$PATH
export LD_LIBRARY_PATH=/home4/intern/stniu/libs/ffmpeg/lib:$LD_LIBRARY_PATH

NCCL_SOCKET_IFNAME=eth0

