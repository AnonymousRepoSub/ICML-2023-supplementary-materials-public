

i=30
algorithms=("cfo" "hypertime")
for algorithm in "${algorithms[@]}"
do  
    for seed in 1 2 3 4 5 #6 7 8 9 42
    do 
        echo "cpu $i: cfo seed$seed"
        taskset -c $i nohup python main.py \
                    --algorithm $algorithm \
                    --dataset 'temp' \
                    --estimator 'lgbm' \
                    --metric 'rmse' \
                    --budget 14400 \
                    --shuffle \
                    --seed $seed &
        i=$(($i + 1))
    done
done





# hypertime
# i=23
# for seed in 1 2 3 4 5 6 7 8 9 42
# do 
#     echo "cpu $i Seed: $seed"

#     taskset -c $i nohup python main_drug.py  --algorithm hypertime --budget 10 --seed $seed &

#     i=$(($i + 1))
# done

# taskset -c 31  python main_weather.py  --algorithm cfo --budget 10 --seed 1 
# taskset -c 25  python main_drug.py  --algorithm cfo --budget 20 --seed 1 &

# to do: run all 5 tolerances sequentially


# hypertime
# i=23
# for seed in 1 2 3 4 5 6 7 8 9 42
# do 
#     echo "cpu $i Seed: $seed"

#     taskset -c $i nohup python main_drug.py  --algorithm hypertime --budget 10 --seed $seed &

#     i=$(($i + 1))
# done

# taskset -c 31  python main_weather.py  --algorithm cfo --budget 10 --seed 1 
# taskset -c 25  python main_drug.py  --algorithm cfo --budget 20 --seed 1 &

# to do: run all 5 tolerances sequentially
# use 11 cpus
# i=29
# echo "cpu $i cfo, $dataset, $estimator, $metric, budget$budget, seed$seed"
# taskset -c $i nohup python test.py \
#                     --algorithm cfo \
#                     --dataset $dataset \ 
#                     --estimator $estimator \ 
#                     --metric $metric \ 
#                     --budget $budget&

# python main.py --algorithm cfo --dataset temp --estimator lgbm --metric rmse --budget 30 --size 50000
                    
