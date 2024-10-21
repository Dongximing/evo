#!/bin/bash

set -ex

export CUBLAS_WORKSPACE_CONFIG=:16:8
export CUDA_VISIBLE_DEVICES=0

BUDGET=10
POPSIZE=10
SEED=5
TEMPLATE=v1
INITIAL_MODE=para_topk
LLM_TYPE=turbo

for task in sports_understanding
do
OUT_PATH=outputs/cls/$DATASET/alpaca/all/de/bd${BUDGET}_top${POPSIZE}_${INITIAL_MODE}_init/${TEMPLATE}/$LLM_TYPE
for SEED in 5 10 15
do
python run.py \
    --seed $SEED \
    --dataset $DATASET \
    --task cls \
    --batch-size 64 \
    --prompt-num 0 \
    --sample_num 500 \
    --language_model alpaca \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --position demon \
    --evo_mode de \
    --llm_type $LLM_TYPE \
    --setting default \
    --write_step 5 \
    --initial all \
    --initial_mode $INITIAL_MODE \
    --template $TEMPLATE \
    --output $OUT_PATH/seed$SEED \
    --cache_path data/cls/$DATASET/seed${SEED}/prompts_batched.json \
    --dev_file ./data/cls/$DATASET/seed${SEED}/dev.txt
done
python get_result.py -p $OUT_PATH > $OUT_PATH/result.txt
done

for task in temporal_sequences
# snarks ruin_names tracking_shuffled_objects_seven_objects tracking_shuffled_objects_five_objects logical_deduction_three_objects hyperbaton logical_deduction_five_objects logical_deduction_seven_objects movie_recommendation salient_translation_error_detection reasoning_about_colored_objects date_understanding multistep_arithmetic_two  navigate  dyck_languages  word_sorting  sports_understanding object_counting  formal_fallacies  causal_judgement  web_of_lies boolean_expressions
do
OUT_PATH=outputs/$task/eval/$llm/3-shot
for seed in 10
do
mkdir -p $OUT_PATH/seed${seed}
python ../eval.py \
    --seed $seed \
    --task $task \
    --batch-size 20 \
    --sample_num 50 \
    --llm_type $llm \
    --setting default \
    --demon 1 \
    --output $OUT_PATH/seed${seed}
done
done