import os
import subprocess

import re


def main(Prefix = '.',
    training_num = 10,
    per_step = 50,
    mode = 0,
    model_mode = "v1.4", 
    checkpointing_steps=1000,
    gen_num=100,
    cfg=5.0,
    fix_term=0.0,
    baseline=False
    ):

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
        
        input_prompt = f"{indicator} {data_type} of {' '.join(re.split(r'[-_]', current_style))}"

        class_dir = f"{Prefix}/datasets_extraction/class_dataset/{src_dataset_path}/{'_'.join(re.split(r'[-_]', current_style))}"

        print(f'Current input prompt is {input_prompt}')

        checkpoint_path = f"{Prefix}/checkpoints/{training_mode_list[mode]}/{src_dataset_path}-{training_num}/{per_step}_{model_mode}/{current_style}/checkpoint-{checkpointing_steps}"
        if baseline:
            output_path = f"{Prefix}/Generator_Output/{src_dataset_path}/baseline_{training_mode_list[mode]}_{training_num}_{per_step}_{model_mode}_cfg_{cfg}/{current_style}"
        else:
            output_path = f"{Prefix}/Generator_Output/{src_dataset_path}/{training_mode_list[mode]}_{training_num}_{per_step}_{model_mode}_cfg_{cfg}_fix_{fix_term}/{current_style}"

        if os.path.exists(f"{output_path}/{gen_num*5-1}.png"):
            print('Gen Already Fin! Skipping.')
            continue
        else:
            print(f'No {output_path}/{gen_num*5-1}.png found. Running Exps')

        prompt = input_prompt
        
        if baseline:
            if mode == 1:
              training_script = f"""
                python test_dreambooth.py \
                --model_path "{checkpoint_path}" \
                --output_path "{output_path}" \
                --gen_num {gen_num} \
                --cfg {cfg} \
                --prompt "{prompt}" \
                --lora
                """
            else:
                training_script = f"""
                python test_dreambooth.py \
                --model_path "{checkpoint_path}" \
                --output_path "{output_path}" \
                --gen_num {gen_num} \
                --cfg {cfg} \
                --prompt "{prompt}"
                """
        else:
            if mode == 1:
                training_script = f"""
                python contrasted_sample.py \
                --model_path "{checkpoint_path}" \
                --output_path "{output_path}" \
                --gen_num {gen_num} \
                --cfg {cfg} \
                --fix_term {fix_term} \
                --prompt "{prompt}" \
                --lora
                """
            else:
                training_script = f"""
                python contrasted_sample.py \
                --model_path "{checkpoint_path}" \
                --output_path "{output_path}" \
                --gen_num {gen_num} \
                --cfg {cfg} \
                --fix_term {fix_term} \
                --prompt "{prompt}"
                """

        subprocess.call(["sh", "-c", training_script])


if __name__ == '__main__':
    main(training_num = 10, mode=0, per_step=200, checkpointing_steps=2000, cfg=3.0, fix_term=-0.02, baseline=False)
    main(training_num = 10, mode=0, per_step=200, checkpointing_steps=2000, cfg=3.0, baseline=True)
    main(training_num = 10, mode=1, per_step=200, checkpointing_steps=2000, cfg=5.0, fix_term=0.0, baseline=False)
    main(training_num = 10, mode=1, per_step=200, checkpointing_steps=2000, cfg=3.0, baseline=True)
