# 1. set the attributes for all 11 cpus!
# 2. manually set tolerances 
# check '\' is placed after every input!!!!!!!!!

# use 11 cpus
# i=0 # init cpu
# algorithms=("cfo" "hypertime" "lexico_var")
# for algorithm in "${algorithms[@]}"
# do  
#     echo "cpu $i "
#     taskset -c $i nohup python test.py \
#                 --algorithm $algorithm \
#                 --dataset 'electricity'\
#                 --estimator 'xgboost'\
#                 --metric 'roc_auc' \
#                 --budget 7200 \
#                 --size 0 &
#     i=$(($i + 1))
# done
# electricity_size0_xgboost_roc_auc_b7200

i=0 # init cpu # "lexico_var" hypertime
algorithms=("cfo" "hypertime")
# for tolerance in 0.005 0.01 0.02 0.04
# do  

# done
for algorithm in "${algorithms[@]}"
do  
    echo "cpu $i $algorithm"
    taskset -c $i nohup python test.py \
                --algorithm $algorithm \
                --dataset 'electricity'\
                --estimator 'xgboost'\
                --metric 'roc_auc' \
                --budget 7200 \
                --size 0 &
    i=$(($i + 1))
    sleep 0.5
done


# python test.py --algorithm cfo \
#                     --dataset 'electricity'\
#                     --estimator 'xgboost'\
#                     --metric 'roc_auc' \
#                     --budget 7200 \
#                     --size 0 

# i=1 # init cpu # "lexico_var"
# algorithms=("cfo")
# for algorithm in "${algorithms[@]}"
# do  
#     echo "cpu $i $algorithm"
#     taskset -c $i nohup python test.py \
#                 --algorithm $algorithm \
#                 --dataset 'vessel'\
#                 --estimator 'xgboost'\
#                 --metric 'rmse' \
#                 --budget 14401 \
#                 --size 0 &
#     i=$(($i + 1))
#     sleep 0.5
# done


# for algorithm in "${algorithms[@]}"
# do  
#     echo "cpu $i $algorithm"
#     taskset -c $i nohup python test.py \
#                 --algorithm $algorithm \
#                 --dataset 'sales'\
#                 --estimator 'lgbm'\
#                 --metric 'rmse' \
#                 --budget 7200 \
#                 --size 0 &
#     i=$(($i + 1))
#     sleep 0.5
# done

# # i=0 # init cpu
# algorithms=("cfo" "hypertime" "lexico_var")
# for algorithm in "${algorithms[@]}"
# do  
#     echo "cpu $i $algorithm"
#     taskset -c $i nohup python test.py \
#                 --algorithm $algorithm \
#                 --dataset 'sales'\
#                 --estimator 'lgbm'\
#                 --metric 'rmse' \
#                 --budget 7200 \
#                 --size 0 &
#     i=$(($i + 1))
#     sleep 0.5
# done

# i=3 # init cpu
# algorithms=("cfo" "hypertime" "lexico_var")
# for algorithm in "${algorithms[@]}"
# do  
#     echo "cpu $i "
#     taskset -c $i nohup python test.py \
#                 --algorithm $algorithm \
#                 --dataset 'temp'\
#                 --estimator 'lgbm'\
#                 --metric 'rmse' \
#                 --budget 28800 \
#                 --size 0 &
#     i=$(($i + 1))
#     sleep 2
# done


# python test.py \
#                 --algorithm hypertime \
#                 --dataset 'temp'\
#                 --estimator 'lgbm'\
#                 --metric 'rmse' \
#                 --budget 28800 \
#                 --size 0 
# python test.py \
#                     --tolerance 0.0057 \
#                     --algorithm hypertime \
#                     --dataset 'drug'\
#                     --estimator 'xgboost'\
#                     --metric 'rmse' \
#                     --budget 10

# taskset -c 38 python test.py \
#                     --algorithm 'cfo' \
#                     --dataset 'temp' \
#                     --estimator 'lgbm'\
#                     --metric 'rmse'\
#                     --budget 14400 \
#                     --size 50000