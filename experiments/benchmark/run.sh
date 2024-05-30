declare -a Envs=("env1" "env2" "env3" "env4")
declare -a Models=("ssgp++" "ovc" "ovc++" "poam")

for seed in {0..9}; do
    for env in ${Envs[@]}; do
        for model in ${Models[@]}; do
            echo $seed $env $model
            mkdir -p ./loginfo/$seed/$env/
            python main.py seed=$seed map=$env model=$model > "./loginfo/${seed}/${env}/${model}.txt"
        done
    done
done
