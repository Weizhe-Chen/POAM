declare -a Models=("poam"  "poam-z-opt"  "poam-z-rand"  "poam-var-opt"  "poam-var-ssgp"  "poam-online-elbo")

for model in ${Models[@]}; do
    echo $model
    mkdir -p ./loginfo/
    python main.py model=$model > "./loginfo/${model}.txt"
done
