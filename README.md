# Node2KG
Transformation of Node to Knowledge Graph Embeddings for Faster Link Prediction in Social Networks 

## System Requirements
Linux


## Setup

1. Clone the repository.  
    `git clone https://github.com/ArchitParnami/Node2KG.git`

2. Create a new conda environment by  
    `conda create -n Node2KG python=3.6`  

3. Activate conda environment  
    `conda activate Node2KG` 

4. Install the required packages  
    `pip install -r requirements.txt`  
 
5. Setup this project   
    `python setup.py develop`  

6. Setup OpenNE (from project root)  
    `cd GraphEmbed/OpenNE` 
    `pip install -r requirements.txt` 
    `cd src` 
    `python setup.py install` 

5. Setup OpenKE (from project root)  
    `cd GraphEmbed/OpenKE` 
    `mkdir release`   
    `./make.sh `  

6. To run the experiments  
    `cd GraphEmbed/scripts`  
    `./run_all.sh`  

