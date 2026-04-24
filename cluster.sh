# Cluster-specific settings for Einsteinium (LRC).
# Source this file in SLURM batch scripts.

# SLURM defaults
CLUSTER_PARTITION_GPU=es1
CLUSTER_PARTITION_CPU=lr6
CLUSTER_ACCOUNT=pc_als831
CLUSTER_QOS_GPU=es_normal
CLUSTER_QOS_CPU=lr_normal
CLUSTER_CPUS_GPU=16        # es_normal minimum
CLUSTER_GRES_GPU="gpu:A40:4"

# Module setup for PyTorch
setup_pytorch() {
    source /etc/profile.d/modules.sh
    export MODULEPATH=$MODULEPATH:/global/software/rocky-8.x86_64/modfiles/Core
    module load --force ml/pytorch/2.3.1-py3.11.7-mf
}

# Module setup for CCP4/gemmi (data generation)
setup_ccp4() {
    source /global/home/groups-sw/ac_als831/ccp4-9/bin/ccp4.setup-sh
}
