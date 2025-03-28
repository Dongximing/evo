#!/bin/bash

set -ex


BUDGET=3
POPSIZE=10
llm=turbo
initial=cot
initial_mode=para_topk
data_method=half_half

#for task in date_understanding dyck_languages  word_sorting  sports_understanding object_counting  formal_fallacies  causal_judgement  web_of_lies temporal_sequences disambiguation_qa tracking_shuffled_objects_three_objects penguins_in_a_table geometric_shapes snarks ruin_names tracking_shuffled_objects_seven_objects tracking_shuffled_objects_five_objects logical_deduction_three_objects hyperbaton logical_deduction_five_objects logical_deduction_seven_objects movie_recommendation salient_translation_error_detection reasoning_about_colored_objects multistep_arithmetic_two navigate boolean_expressions
for task in metaphor_boolean
do
for SIZE in 10
do
POPSIZE=$SIZE
OUT_PATH=outputs/$task/$initial/ga/bd${BUDGET}_top${POPSIZE}_${initial_mode}_init/$llm/$data_method
for seed in 100 200 300
do
mkdir -p $OUT_PATH/seed${seed}
cache_path=cache/$task/seed$seed
mkdir -p $cache_path
python ../run.py \
    --seed $seed \
    --task $task \
    --batch-size 1 \
    --sample_num 20 \
    --budget $BUDGET \
    --popsize $POPSIZE \
    --evo_mode ga \
    --llm_type $llm \
    --setting default \
    --initial $initial \
    --initial_mode $initial_mode \
    --cot_cache_path $cache_path/prompts_cot_$llm$data_method$seed.json \
    --output $OUT_PATH/seed${seed}\
    --sampling_method  $data_method

done
done
done