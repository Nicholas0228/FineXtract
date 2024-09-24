import os
import subprocess
import re



def main(Prefix='.',
    training_num = 10,
    per_step = 50,
    mode = 0,
    model_mode = "v1.4", 
    checkpointing_steps=1000):

    training_mode_list = ['db', 'lora']
    
    overall_step = training_num * per_step
    src_dataset_path = "demo-wikiart"
    data_type = "style"
    indicator = 'art'
    if mode == 0:
        suffix = ''
    elif mode == 1:
        suffix = '_lora'
    instance_src = f"{Prefix}/datasets_extraction/{src_dataset_path}-{training_num}{suffix}"

    for current_style in os.listdir(instance_src):
        if current_style.endswith('.txt'):
            print(f'Passing {current_style}')
            continue
        instance_dir = f"{instance_src}/{current_style}"
        output_dir = f"{Prefix}/checkpoints/{training_mode_list[mode]}/{src_dataset_path}-{training_num}/{per_step}_{model_mode}/{current_style}"


        input_prompt = f"{indicator} {data_type} of {' '.join(re.split(r'[-_]', current_style))}"

        class_dir = f"{Prefix}/datasets_extraction/class_dataset/{src_dataset_path}/{'_'.join(re.split(r'[-_]', current_style))}"

        print(f'Current input prompt is {input_prompt}')

        if os.path.exists(f"{output_dir}/checkpoint-{checkpointing_steps}"):
            print(f"skip exist output dir {output_dir}")
            continue
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
      
        if mode == 0:
            training_script = f"""
            export MODEL_NAME="CompVis/stable-diffusion-v1-4"
            export INSTANCE_DIR="{Prefix}/datasets_extraction/{src_dataset_path}-{training_num}{suffix}/{current_style}"
            export OUTPUT_DIR="{Prefix}/checkpoints/{training_mode_list[mode]}/{src_dataset_path}-{training_num}/{per_step}_{model_mode}/{current_style}"
            accelerate launch train_dreambooth.py \
            --pretrained_model_name_or_path=$MODEL_NAME  \
            --instance_data_dir=$INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --instance_prompt="{input_prompt}" \
            --resolution=512 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=1 \
            --learning_rate=2e-6 \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --max_train_steps={overall_step} \
            --checkpointing_steps={checkpointing_steps} 
            """
        elif mode == 1:
            training_script = f"""
            export MODEL_NAME="CompVis/stable-diffusion-v1-4"
            export INSTANCE_DIR="{Prefix}/datasets_extraction/{src_dataset_path}-{training_num}{suffix}/{current_style}"
            export OUTPUT_DIR="{Prefix}/checkpoints/{training_mode_list[mode]}/{src_dataset_path}-{training_num}/{per_step}_{model_mode}/{current_style}"
            export CHECKPOINT_STEP="{checkpointing_steps}"
            export TRAINING_STEPS="{overall_step}"
            accelerate launch train_text2img_lora.py \
                --pretrained_model_name_or_path=$MODEL_NAME  \
                 --dataset_name=$INSTANCE_DIR \
                 --caption_column="caption" \
                --resolution=512 \
                --output_dir=$OUTPUT_DIR \
                --resolution=512 \
                --train_batch_size=1 \
                --checkpointing_steps=$CHECKPOINT_STEP \
                --learning_rate=1e-4 \
                --lr_scheduler="constant" \
                --max_train_steps=$TRAINING_STEPS \
                --lr_warmup_steps=0 \
                --rank=64 \
                
            """
        else:
            raise ValueError('Not Implemented')
        subprocess.call(["sh", "-c", training_script])



if __name__ == '__main__':
    main(training_num = 10, mode=0, per_step=200, checkpointing_steps=2000)
    main(training_num = 10, mode=1, per_step=200, checkpointing_steps=2000)