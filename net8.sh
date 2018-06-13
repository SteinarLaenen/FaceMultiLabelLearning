#!/usr/bin/env bash
cd "$(dirname "$0")"



## RUN1
echo split0run1 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --resume checkpoint_run1split0net8_lr00005wd3.pth.tar --filename run1split0net8_lr00005wd3 --seed 100 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run1 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --resume checkpoint_run1split1net8_lr00005wd3.pth.tar --filename run1split1net8_lr00005wd3 --seed 101 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run1 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run1split2net8_lr00005wd3 --seed 102 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun1 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run1fullnet8compare_lr00005wd3 --seed 103 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

RUN2
echo split0run2 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run2split0net8_lr00005wd3 --seed 104 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&


echo split1run2 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run2split1net8_lr00005wd3 --seed 105 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

    
echo split2run2 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run2split2net8_lr00005wd3 --seed 106 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun2 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run2fullnet8compare_lr00005wd3 --seed 107 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

## RUN3
echo split0run3 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --resume checkpoint_run3split0net8_lr00005wd3.pth.tar --filename run3split0net8_lr00005wd3 --seed 108 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run3 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --resume checkpoint_run3split1net8_lr00005wd3.pth.tar --filename run3split1net8_lr00005wd3 --seed 109 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run3 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run3split2net8_lr00005wd3 --seed 110 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun3 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run3fullnet8_lr00005wd3 --seed 111 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72


    ## RUN4

    
echo split0run4 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run4split0net8_lr00005wd3 --seed 111 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run4 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run4split1net8_lr00005wd3 --seed 112 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run4 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run4split2net8_lr00005wd3 --seed 113 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun4 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run4fullnet8_lr00005wd3 --seed 114 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72


## RUN5
echo split0run5 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run5split0net8_lr00005wd3 --seed 115 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run5 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run5split1net8_lr00005wd3 --seed 116 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run5 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run5split2net8_lr00005wd3 --seed 117 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun5 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run5fullnet8_lr00005wd3 --seed 118 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72


## RUN6
echo split0run6 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run6split0net8_lr00005wd3 --seed 119 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run6 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run6split1net8_lr00005wd3 --seed 120 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run6 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run6split2net8_lr00005wd3 --seed 121 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun6 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run6fullnet8_lr00005wd3 --seed 122 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

## RUN7
echo split0run7 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run7split0net8_lr00005wd3 --seed 123 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run7 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run7split1net8_lr00005wd3 --seed 124 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run7 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run7split2net8_lr00005wd3 --seed 125 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun7 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run7fullnet8_lr00005wd3 --seed 126 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72


## RUN8
echo split0run8 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run8split0net8_lr00005wd3 --seed 127 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run8 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run8split1net8_lr00005wd3 --seed 128 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run8 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run8split2net8_lr00005wd3 --seed 129 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun8 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run8fullnet8_lr00005wd3 --seed 130 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72



## RUN9
echo split0run9 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run9split0net8_lr00005wd3 --seed 131 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run9 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run9split1net8_lr00005wd3 --seed 132 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split2run9 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run9split2net8_lr00005wd3 --seed 133 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun9 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run9fullnet8_lr00005wd3 --seed 134 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72


## RUN10
echo split0run10 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split0 --filename run10split0net8_lr00005wd3 --seed 135 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72 &&

echo split1run10 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split1 --filename run10split1net8_lr00005wd3 --seed 136 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo split2run10 &&
    python /media/SixTB/steinar/Models/final/facedatasplit/facedatasplit.py -a multitaskresnet8 --split split2 --filename run10split2net8_lr00005wd3 --seed 137 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72

echo fullrun10 &&
    python /media/SixTB/steinar/Models/final/facedatafull/facedatafull.py -a resnet8 --filename run10fullnet8_lr00005wd3 --seed 138 -b 400 -j 6 --lr 0.0005 --wd 1e-3 --epochs 30 /media/SixTB/steinar/Data/facedatarandom_72
