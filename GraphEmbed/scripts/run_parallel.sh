# GLOBALS
limit=-1
dim=32
OpenKE_epochs=200
FineTune_epochs=200
model='self_attention'
compare_split='test'
patience=10

datasets=("youtube" "dblp" "amazon" "lj" "orkut" "orkut" "orkut")
dataset_min_nodes=(16 16 21 61 151 251 351)
dataset_max_nodes=(20 20 25 65 155 255 355)
dataset_random_walks=(10 10 10 20 20 25 25)
dataset_walk_size=(5 5 8 10 20 25 25)
batch_sizes=(32 32 32 32 32 64 64)
dataset_loss=("euclidean" "euclidean" "euclidean" "euclidean" "mse" "cosine" "cosine")
dataset_max_epochs=(12 12 12 12 20 100 200) 

OpenNE_methods=("node2vec" "deepWalk")
OpenKE_methods=("TransE" "TransH" "TransD" "SimplE" "RESCAL" "DistMult")
plot_kind='box'

#Conditionals to control process executions
create_dataset=true
create_openne=true
create_openke=true
create_openne_openke=true
create_transform=true
do_train=true
do_transform=true
do_evaluate=true
eval_openne=true
eval_openke=true
eval_openne_openke=true
eval_transformed=true
do_compare=true

eval_embeddings()
{
    
    dataset=$1; min_size=$2; max_size=$3; OpenKE=$4; OpenNE=$5; embedding=$6

    folder=$(date +%Y-%m-%d--%H:%M:%S)
    result_dir=results/$dataset/$min_size-$max_size/$dim/$OpenNE-$OpenKE/$1/$folder
    mkdir -p $result_dir
    python -W ignore eval_embeddings.py  --dataset=$dataset --min_size=$min_size --max_size=$max_size --dim=$dim \
    --eval_method=$OpenKE --embedding_type=$embedding --openne_type=$OpenNE --result_path=$result_dir >  $result_dir/results.txt
    python parse_results.py --path=$result_dir
    #rm -rf results
}

download_dataset()
{
    dataset=$1
    echo "Downloading $dataset dataset"
    python download_dataset.py --dataset=$dataset
}

build_dataset()
{
    dataset=$1; min_size=$2; max_size=$3
    
    echo "Building dataset $dataset/$min_size-$max_size"
    python -W ignore build_dataset.py --dataset=$dataset --min_size=$min_size --max_size=$max_size --num_graphs=$limit
}


build_openne()
{
    dataset=$1; min_size=$2; max_size=$3; OpenNE=$4; random_walks=$5; walk_size=$6
    
    echo "Generating $OpenNE embeddings for $dataset/$min_size-$max_size"
    python -W ignore build_embeddings_OpenNE.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --number-walks=$random_walks --walk-length=$walk_size --representation-size=$dim --method=$OpenNE
}

build_openke()
{
    dataset=$1; min_size=$2; max_size=$3; OpenKE=$4
    
    echo "Generating $OpenKE embeddings for $dataset/$min_size-$max_size"
    python -W ignore build_embeddings_OpenKE.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --train_method=$OpenKE --epochs=$OpenKE_epochs --dim=$dim >> LOG.txt
}


build_openke_from_openne()
{
    dataset=$1; min_size=$2; max_size=$3; OpenKE=$4; OpenNE=$5

    echo "Generating $OpenKE embeddings from $OpenNE for $dataset/$min_size-$max_size"
    python -W ignore build_embeddings_OpenKE.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --train_method=$OpenKE --use_openne --openne_embedding=$OpenNE --epochs=$FineTune_epochs --dim=$dim >> LOG.txt

}

split_dataset()
{
    dataset=$1; min_size=$2; max_size=$3
    split_name=$dataset'-'$min_size'-'$max_size

    echo "Spliting $dataset dataset into train, validation and test sets"
    python make_split.py --dataset=$dataset --min_size=$min_size --max_size=$max_size --name=$split_name --shuffle
}

train()
{
    dataset=$1; min_size=$2; max_size=$3; batch_size=$4; OpenNE=$5; OpenKE=$6; model_save_name=$7; epochs=$8; loss_fun=$9
    split_name=$dataset'-'$min_size'-'$max_size
    
    echo "Training the transformation model using $model for $dataset/$min_size-$max_size. Source=$OpenNE Target=$OpenKE"
    python train.py --dataset=$dataset --min_size=$min_size --max_size=$max_size --split_name=$split_name \
    --dim=$dim --batch_size=$batch_size --model=$model --epochs=$epochs --source=$OpenNE --target=$OpenKE \
    --name=$model_save_name --loss=$loss_fun --patience=$patience
}


transform()
{
    dataset=$1; min_size=$2; max_size=$3;OpenNE=$4; OpenKE=$5; model_save_name=$6

    echo "Transforming Embeddings for $dataset/$min_size-$max_size"
    python transform_embeddings.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --source=$OpenNE --target=$OpenKE --model_dir=$model_save_name --dim=$dim
}

evaluate()
{
    dataset=$1; min_size=$2; max_size=$3; OpenKE=$4; OpenNE=$5
    
    echo "Evaluating.."
    
    if [ "$eval_openne" = true ]
    then
        echo "  (1) Evaluating $OpenNE embeddings"
        eval_embeddings $dataset $min_size $max_size $OpenKE $OpenNE openne
    fi

    if [ "$eval_openke" = true ]
    then
        echo "  (2) Evaluating $OpenKE embeddings"
        eval_embeddings  $dataset $min_size $max_size $OpenKE $OpenNE openke
    fi

    if [ "$eval_openne_openke" = true ]
    then
        echo "  (3) Evaluating Finetuned ($OpenNE->$OpenKE) embeddings"
        eval_embeddings $dataset $min_size $max_size $OpenKE $OpenNE openne_openke
    fi

    if [ "$eval_transformed" = true ]
    then
        echo "  (4) Evaluating Transformed ($OpenNE -> $model) embeddings"
        eval_embeddings  $dataset $min_size $max_size $OpenKE $OpenNE transformed
    fi
}

compare()
{
    dataset=$1; min_size=$2; max_size=$3; OpenKE=$4; OpenNE=$5
    split_name=$dataset'-'$min_size'-'$max_size

    echo "Comparing Results for  $dataset/$min_size-$max_size"
    python compare.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --source=$OpenNE --target=$OpenKE --split_name=$split_name --plot_split=$compare_split --plot=$plot_kind \
    --dim=$dim
}


process_transform()
{
    dataset=$1; min_size=$2; max_size=$3; batch_size=$4; OpenNE=$5; OpenKE=$6; epochs=$7; loss_fun=$8
    model_save_name=$(date +%Y-%m-%d--%H:%M:%S)
    
    if [ "$create_openne_openke" = true ]
    then
        build_openke_from_openne $dataset $min_size $max_size $OpenKE $OpenNE    
    fi
    
    if [ "$do_train" = true ]
    then
        train $dataset $min_size $max_size $batch_size $OpenNE $OpenKE $model_save_name $epochs $loss_fun
    fi
    
    if [ "$do_transform" = true ]
    then
        transform $dataset $min_size $max_size $OpenNE $OpenKE $model_save_name
    fi
  
    if [ "$do_evaluate" = true ]
    then
        evaluate $dataset $min_size $max_size $OpenKE $OpenNE
    fi
    
    if [ "$do_compare" = true ]
    then
        compare $dataset $min_size $max_size $OpenKE $OpenNE
    fi
}

process_dataset()
{
    dataset=$1; min_size=$2; max_size=$3; batch_size=$4; random_walks=$5; walk_size=$6; epochs=$7; loss_fun=$8

    if [ "$create_dataset" = true ]
    then
        
        download_dataset $dataset
        build_dataset $dataset $min_size $max_size
        split_dataset $dataset $min_size $max_size
    fi

    if [ "$create_openne" = true ]
    then
        I=0    
        for source_method in ${OpenNE_methods[@]}
        do
            echo -e "\n===SOURCE: $(expr $I + 1) of ${#OpenNE_methods[@]}===\n"
            build_openne  $dataset $min_size $max_size $source_method $random_walks $walk_size &
            I=$(expr $I + 1)
        done
    fi
    
    if [ "$create_openke" = true ]
    then
        J=0
        for target_method in ${OpenKE_methods[@]}
        do
            echo -e "\n===TARGET: $(expr $J + 1) of ${#OpenKE_methods[@]}===\n"
            build_openke $dataset $min_size $max_size $target_method &
            J=$(expr $J + 1)
        done
    fi

    wait

    K=0
    for source_method in ${OpenNE_methods[@]}
    do
        for target_method in ${OpenKE_methods[@]}
        do
            echo -e "\n===TRANSFORM: $(expr $K + 1) of $(expr ${#OpenNE_methods[@]} \* ${#OpenKE_methods[@]})===\n"
            process_transform $dataset $min_size $max_size $batch_size $source_method $target_method $epochs $loss_fun &
            K=$(expr $K + 1)
        done
    done

    wait
}


main()
{
    ITER=0
    for dataset_name in ${datasets[@]}
    do
        echo -e "\n===DATASET: $(expr $ITER + 1) of ${#datasets[@]}===\n"
        dataset=$dataset_name
        min_size=${dataset_min_nodes[$ITER]}
        max_size=${dataset_max_nodes[$ITER]}
        batch_size=${batch_sizes[$ITER]}
        random_walks=${dataset_random_walks[$ITER]}
        walk_size=${dataset_walk_size[$ITER]}
        epochs=${dataset_max_epochs[$ITER]}
        loss_fun=${dataset_loss[$ITER]}
        process_dataset $dataset $min_size $max_size $batch_size $random_walks $walk_size $epochs $loss_fun &
        
        ITER=$(expr $ITER + 1)
    done

    wait

    echo "Finished."
}

main
