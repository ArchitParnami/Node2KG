
# aurguments: 
    # $1 -> embedding_type : [openne, openke, openne-openke, transformed]
eval_embeddings()
{
    folder=$(date +%Y-%m-%d--%H:%M:%S)
    result_dir=results/$dataset/$min_size-$max_size/$OpenNE-$OpenKE/$1/$folder
    mkdir -p $result_dir
    python -W ignore eval_embeddings.py  --dataset=$dataset --min_size=$min_size --max_size=$max_size --dim=$dim \
    --eval_method=$OpenKE --embedding_type=$1 --openne_type=$OpenNE --result_path=$result_dir >  $result_dir/results.txt
    python parse_results.py --path=$result_dir
    #rm -rf results
}

build_dataset()
{
    echo "Building dataset $dataset/$min_size-$max_size"
    python -W ignore build_dataset.py --dataset=$dataset --min_size=$min_size --max_size=$max_size --num_graphs=$limit
}


build_openne()
{
    echo "Generating $OpenNE embeddings for $dataset/$min_size-$max_size"
    python -W ignore build_embeddings_OpenNE.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --number-walks=$random_walks --walk-length=$walk_size --representation-size=$dim --method=$OpenNE
}

build_openke()
{
    echo "Generating $OpenKE embeddings for $dataset/$min_size-$max_size"
    python -W ignore build_embeddings_OpenKE.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --train_method=$OpenKE --epochs=$OpenKE_epochs --dim=$dim >> LOG.txt
}


build_openke_from_openne()
{
    echo "Generating $OpenKE embeddings from $OpenNE for $dataset/$min_size-$max_size"
    python -W ignore build_embeddings_OpenKE.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --train_method=$OpenKE --use_openne --openne_embedding=$OpenNE --epochs=$FineTune_epochs --dim=$dim >> LOG.txt

}

split_dataset()
{
    echo "Spliting $dataset dataset into train, validation and test sets"
    python make_split.py --dataset=$dataset --min_size=$min_size --max_size=$max_size --name=$split_name
}

train()
{
    echo "Training the transformation model using $model for $dataset/$min_size-$max_size. Source=$OpenNE Target=$OpenKE"
    python train.py --dataset=$dataset --min_size=$min_size --max_size=$max_size --split_name=$split_name \
    --dim=$dim --batch_size=$batch_size --model=$model --epochs=$epochs --source=$OpenNE --target=$OpenKE \
    --name=$model_save_name
}


transform()
{
    echo "Transforming Embeddings for $dataset/$min_size-$max_size"
    python transform_embeddings.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --source=$OpenNE --target=$OpenKE --model_dir=$model_save_name
}

evaluate()
{
    echo "Evaluating.."
    echo "  (1) Evaluating $OpenNE embeddings"
    eval_embeddings openne
    echo "  (2) Evaluating $OpenKE embeddings"
    eval_embeddings openke
    echo "  (3) Evaluating Finetuned ($OpenNE->$OpenKE) embeddings"
    eval_embeddings openne_openke
    echo "  (4) Evaluating Transformed ($OpenNE -> $model) embeddings"
    eval_embeddings transformed
}

compare()
{
    echo "Comparing Results for  $dataset/$min_size-$max_size"
    python compare.py --dataset=$dataset --min_size=$min_size --max_size=$max_size \
    --source=$OpenNE --target=$OpenKE --split_name=$split_name --plot_split=$compare_split
}


#dataset
dataset=''
min_size=0
max_size=0
limit=-1

#OpenNE
random_walks=10
walk_size=80
dim=32
OpenNE=''

#OpenKE
OpenKE=''
OpenKE_epochs=200
FineTune_epochs=200

#train
batch_size=0
model='self_attention'
epochs=8
compare_split='test'

#Conditionals
create_dataset=true
create_openne=true
create_openke=true
create_openne_openke=true
create_transform=true
do_train=true
do_transform=true
do_evaluate=true
do_compare=true

main()
{
    datasets=("amazon" "youtube" "dblp")
    dataset_min_nodes=(21 16 16)
    dataset_max_nodes=(25 20 20)
    batch_sizes=(64 8 8)
    OpenNE_methods=("node2vec" "deepWalk")
    OpenKE_methods=("TransE" "TransH" "TransD" "SimplE" "RESCAL" "DistMult")
    
    ITER=0
    for dataset_name in ${datasets[@]}
    do
        echo -e "\n===DATASET: $(expr $ITER + 1) of ${#datasets[@]}===\n"
        dataset=$dataset_name
        min_size=${dataset_min_nodes[$ITER]}
        max_size=${dataset_max_nodes[$ITER]}
        batch_size=${batch_sizes[$ITER]}
        split_name=$dataset'-'$min_size'-'$max_size
        
        if [ "$create_dataset" = true ]
        then    
            build_dataset
            split_dataset
        fi
        
        I=0
        for source_method in ${OpenNE_methods[@]}
        do
            echo -e "\n===SOURCE: $(expr $I + 1) of ${#OpenNE_methods[@]}===\n"
            OpenNE=$source_method
            
            if [ "$create_openne" = true ]
            then
                build_openne
            fi

            J=0
            for target_method in ${OpenKE_methods[@]}
            do
                echo -e "\n===TARGET: $(expr $J + 1) of ${#OpenKE_methods[@]}===\n"
                OpenKE=$target_method
                
                if [ "$create_openke" = true ] && [ $I -eq 0 ] #run only once
                then
                    build_openke
                fi
                
                if [ "$create_openne_openke" = true ]
                then
                    build_openke_from_openne
                fi

                model_save_name=$(date +%Y-%m-%d--%H:%M:%S)
                if [ "$do_train" = true ]; then train; fi
                if [ "$do_transform" = true ]; then transform; fi
                if [ "$do_evaluate" = true ]; then evaluate; fi
                if [ "$do_compare" = true ]; then compare; fi
                J=$(expr $J + 1)
            done
            I=$(expr $I + 1)
        done
        ITER=$(expr $ITER + 1)
    done
}

main
