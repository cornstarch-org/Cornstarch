#!/bin/bash

# Run the scaling laws example
# This script will run the scaling laws example with the default parameters


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 1 --rank-llm 32 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 4 --rank-llm 32 --data-scale $i  
done

for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 8 --rank-llm 32 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 16 --rank-llm 32 --data-scale $i  
done

for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 32 --rank-llm 32 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 64 --rank-llm 32 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 1 --rank-llm 64 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 4 --rank-llm 64 --data-scale $i  
done

for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 8 --rank-llm 64 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 16 --rank-llm 64 --data-scale $i  
done

for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 32 --rank-llm 64 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 64 --rank-llm 64 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 1 --rank-llm 16 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 4 --rank-llm 16 --data-scale $i  
done

for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 8 --rank-llm 16 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 16 --rank-llm 16 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 32 --rank-llm 16 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 64 --rank-llm 16 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 1 --rank-llm 8 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 4 --rank-llm 8 --data-scale $i  
done

for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 8 --rank-llm 8 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 16 --rank-llm 8 --data-scale $i  
done

for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 32 --rank-llm 8 --data-scale $i  
done


for i in $(seq 4 12)
do
    echo "START data-scale:" $i
    python finetune_on_llava_sc2.py --model-size "0.5B" --rank-ve 64 --rank-llm 8 --data-scale $i  
done
