#!/bin/bash
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 16
#SBATCH --mem=50g
#SBATCH --gres=gpu:A40:1
#SBATCH -p qTRDGPU
#SBATCH -t 4-00
#SBATCH -J wirehead-generate
#SBATCH -error%A-%a.err
#SBATCH -A psy53c17
sleep 10s
echo $HOSTNAME >&2
source /data/users1/mdoan4/anaconda3/etc/profile.d/conda.sh
conda activate wirehead
python /data/users1/mdoan4/wirehead/src/generate.py $SLURM_ARRAY_TASK_ID
wait
mdoan4@arctrdlogin001:~/wirehead$ cat launch_wirehead.sh
#!/bin/bash

# List of Slurm scripts to execute
declare -a scripts=("wirehead_generator.sh")

# Loop to execute each script
for script in "${scripts[@]}"; do
    sbatch $script
done


