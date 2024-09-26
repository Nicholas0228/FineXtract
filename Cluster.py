# Time Costs: ~3min per running for extraction; ~12min per running for finding cliques. -> 15min per running
import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm
import copy
import torch.nn.functional as F
import networkx as nx
import shutil
from networkx.algorithms.approximation import clique
import re
from torchvision import transforms
import threading
import multiprocessing

max_wait_time = 30
Prefix = '.'
global_type='sscd'
sscd_model = torch.jit.load(f"{Prefix}/pretrainedmodels/sscd_disc_large.torchscript.pt").cuda()


@torch.no_grad()
def extract_features_sscd(model, samples):
    samples = samples.cuda()
    feats = model(samples).clone()
    return feats


def load_image(image_path):
    with Image.open(image_path) as img:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize to ensure all images are the same size
            transforms.ToTensor(),  # Convert to tensor
        ])
        return transform(img)
    


def get_feature(input_image, model_type='sscd'):
    if model_type == 'sscd':
        ret_transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])
        
        img_src = Image.open(input_image)
        if img_src.mode == 'RGBA':
            img_src = img_src.convert('RGB')
        # print(input_image)
        src_tensor = ret_transform(img_src).unsqueeze(0)
        image_feature = extract_features_sscd(sscd_model, src_tensor)
    return image_feature


def measure_extracted_untargeted(extract_path, src_path, save_path, metric_type='dino', compute_type = 'AS+', max_count=None):
    
    def l2_metric(tensor1, tensor2):
        dimensions = 1
        for k in tensor1.shape:
            dimensions *= k
        return torch.sqrt(torch.sum((tensor1 - tensor2)**2/(dimensions)))
    
    def cosine_metric(tensor1, tensor2):
        return 1 - torch.nn.functional.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0).item()
 
    if metric_type == 'L2':
        metric_function = l2_metric
    elif (metric_type == 'dino')  or (metric_type == 'clip') or (metric_type == 'sscd'):
        metric_function = cosine_metric
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    overall_distance = 0.0
    count = 0
    if compute_type == 'AS-':
        for src_root, _, src_files in os.walk(src_path):
            for src_filename in src_files:
                if src_filename.endswith('.csv'):
                    continue
                src_img_path = os.path.join(src_root, src_filename)
                if metric_type == 'dino':
                    src_img_tensor = get_feature(src_img_path, 'dino')  # 提取图像特征
                elif metric_type == 'sscd':
                    src_img_tensor = get_feature(src_img_path, 'sscd')  # 提取图像特征
                else:
                    src_img_tensor = load_image(src_img_path)  # 提取图像特征
                nearest_distance = float('inf')
                nearest_img_path = None
                nearest_img_tensor = None
                subcount = 0
                for root, _, files in os.walk(extract_path):
                    for filename in files:
                        subcount += 1
                        extract_img_path = os.path.join(root, filename)
                        if metric_type == 'dino':
                            extract_img_tensor = get_feature(extract_img_path, 'dino')  # 提取图像特征
                        elif metric_type == 'sscd':
                            extract_img_tensor = get_feature(extract_img_path, 'sscd')  # 提取图像特征
                        elif metric_type == 'L2':
                            extract_img_tensor = load_image(extract_img_path)  # 提取图像特征
                        # 在 src_path 里面找一个离该图片最近的图
            
                        distance = metric_function(extract_img_tensor, src_img_tensor)
                        
                        if distance < nearest_distance:
                            nearest_distance = distance
                            # nearest_img_path = src_img_path
                            nearest_img_path = extract_img_path
                            # nearest_img_tensor = src_img_tensor
                            nearest_img_tensor = extract_img_tensor
                        if max_count is not None:
                            if subcount > max_count - 1:
                                break

                relative_path = os.path.relpath(src_img_path, src_root)
    
                save_dir = os.path.join(save_path, os.path.basename(extract_path), f"{metric_type}_{compute_type}", os.path.splitext(relative_path)[0])
                # print(nearest_img_path, os.path.basename(extract_path), extract_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with open(os.path.join(save_dir, 'distance.txt'), 'w') as f:
                    f.write(str(nearest_distance))
                overall_distance += nearest_distance
                count += 1

                
                shutil.copyfile(src_img_path, os.path.join(save_dir, 'src_img.jpg'))
                shutil.copyfile(nearest_img_path, os.path.join(save_dir, 'nearest_ext_img.jpg'))

    print(overall_distance/count)
    with open(os.path.join(f"{save_path}/{os.path.basename(extract_path)}", f'distance_{metric_type}_{compute_type}.txt'), 'w') as f:
        f.write(str(overall_distance/count))


def run_with_timeout(func, args=(), kwargs={}, timeout_duration=1):
    def target_func(q):
        result = func(*args, **kwargs)
        q.put(result)
    
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=target_func, args=(q,))
    p.start()
    p.join(timeout_duration)
    
    if p.is_alive():
        p.terminate()  # 超时后终止进程
        p.join()
        print(f"Function execution exceeded {timeout_duration} seconds. Moving to the next iteration.")
        return None
    else:
        return q.get() if not q.empty() else None


def find_large_clique(G):
    large_clique = clique.max_clique(G)
    return large_clique


@ torch.no_grad()
def load_images_to_tensor(directory, num_images):
    tensors = []
    img_name_list = []
    
    for idx, filename in tqdm(enumerate(os.listdir(directory))):
        if idx >= num_images:
            break
        path = os.path.join(directory, filename)
        img_name_list.append(path)
        if global_type=='dino':
            tensor = get_feature(path, model_type='dino')
        elif global_type=='sscd':
            tensor = get_feature(path, model_type='sscd')
        else:
            tensor = get_feature(path)
        tensors.append(tensor.detach().cpu())  # Add batch dimension
  
    return torch.cat(tensors), img_name_list  # Shape: [num_images, 4, 512, 512]


@ torch.no_grad()
def clip_sc(feature1, feature2):
    return F.cosine_similarity(feature1[:, None, :], feature2[None, :, :], dim=2)



@ torch.no_grad()
def find_large_cliques_reduced(max_distances, threshold, min_clique_size_list=None):
    num_images = max_distances.shape[0]
    G = nx.Graph()

    # Add nodes for each image
    G.add_nodes_from(range(num_images))

    # Add edges based on the threshold distance
    for i in range(num_images):
        for j in range(i + 1, num_images):  # Avoid self-loops and redundant edges
            if max_distances[i, j] >= threshold:
                G.add_edge(i, j)
    original_G = copy.deepcopy(G) 
    
    return_list = []
    for min_clique_size in min_clique_size_list:
        G = copy.deepcopy(original_G)
        max_cliques = []
        while G.number_of_nodes() > 0:
            # Find all maximal cliques
            cliques = list(nx.find_cliques(G))
            # Filter cliques to find the largest one in this iteration
            if cliques:
                largest_clique = max(cliques, key=len)
                if len(largest_clique) >= min_clique_size:
                    max_cliques.append(largest_clique)
                    # Remove all nodes in the largest clique from the graph
                    G.remove_nodes_from(largest_clique)
                else:
                    break
            else:
                break
        return_list.append(max_cliques)
    # print(return_list)
    return return_list



def find_cluster(src_path, tar_path, num_images, per_img_num, countnum=1, clique_size_list=None, sim_type='dino'):
    global global_type
    global_type = sim_type
    threshold = 1.00
    if clique_size_list is None:
        clique_size_list = [5]
     
    images_tensor, img_name_list = load_images_to_tensor(src_path, num_images)
    print(images_tensor.shape)
    distances = clip_sc(images_tensor, images_tensor)
    for i in range(distances.shape[0]):
        distances[i][i] *= 0
    flattened_distances = distances.view(-1).to(dtype=torch.float32)
    max_values, indices = torch.topk(flattened_distances, 50, largest=True)
    print(max_values)
    if max(max_values) == 0.0:
        raise ValueError()
    threshold = int(max(max_values)/0.01) * 0.01

    while threshold != 0:
        threshold -= 0.01
        cliques = find_large_cliques_reduced(distances, threshold=threshold, min_clique_size_list=clique_size_list)
        count = 0
        stopflag = False
        for clique_size in clique_size_list:
            subcount = 0
            print(f"{threshold:.2f}_{clique_size}: {len(cliques[count])}")
            saved_path = os.path.join(tar_path, f"{threshold:.2f}_{clique_size}")
            for cliques_set in cliques[count]:
                saved_path_dir = os.path.join(saved_path, str(cliques_set[0]))
                # print(saved_path_dir)
                if not os.path.exists(saved_path_dir):
                    os.makedirs(saved_path_dir)
                for clique_id in cliques_set:
                    img_id = img_name_list[clique_id]
                    tar_img = os.path.join(saved_path_dir, img_id.split('/')[-1])
                    # print(img_id, tar_img)
                    shutil.copyfile(img_id, tar_img)
                subcount += 1
            if subcount >= countnum:
                stopflag = True
            count += 1
        if stopflag:
            break


def cluster_progress(src_path, num_images, tar_path, training_num=10):
    images_tensor, img_name_list = load_images_to_tensor(src_path, num_images)
    print(images_tensor.shape)
    distances = clip_sc(images_tensor, images_tensor)
    for i in range(distances.shape[0]):
        distances[i][i] *= 0
    flattened_distances = distances.view(-1).to(dtype=torch.float32)
    max_values, indices = torch.topk(flattened_distances, 50, largest=True)
    min_values, min_indices = torch.topk(flattened_distances, 50, largest=False)

    # threshold = (torch.quantile(flattened_distances, 0.25))//0.01 * 0.01
    # Downsample by randomly selecting 1000 values
    print(max_values, min_values)
    sampled_distances = flattened_distances[torch.randperm(flattened_distances.size(0))[:1000]]
    threshold = torch.quantile(sampled_distances, 0.25) // 0.01 * 0.01


    clique_size_list = [1]
    # threshold =  0.05
    # threshold =  0.05

    while 1:
        cliques = run_with_timeout(find_large_cliques_reduced, args=(distances, threshold, clique_size_list), timeout_duration=max_wait_time)
        print(cliques, threshold)
        if cliques is None:
            threshold += 0.01
            continue

        # cliques = find_large_cliques_reduced(distances, threshold=threshold, min_clique_size_list=clique_size_list)
        count = 0
        stopflag = False
        print(threshold, len(cliques[0]))
        for clique_num in cliques[0]:
            print(f'size {len(clique_num)}')
        threshold += 0.01
        if len(cliques[0]) >= training_num:
            for clique_size in clique_size_list:
                print(f"{threshold:.2f}_{clique_size}: {len(cliques[count])}")
                saved_path = os.path.join(tar_path, f"{threshold:.2f}_{clique_size}")
                for cliques_set in cliques[count]:
                    saved_path_dir = os.path.join(saved_path, str(cliques_set[0]))
                    # print(saved_path_dir)
                    if not os.path.exists(saved_path_dir):
                        os.makedirs(saved_path_dir)
                    for clique_id in cliques_set:
                        img_id = img_name_list[clique_id]
                        tar_img = os.path.join(saved_path_dir, img_id.split('/')[-1])
                        # print(img_id, tar_img)
                        shutil.copyfile(img_id, tar_img)
            break
        if threshold > 1:
            break
        continue


def find_average_id(threshold_dir_sub, metric_type):
    img_name_list = []
    feature_list = []
    for k in os.listdir(threshold_dir_sub):
        img_name = os.path.join(threshold_dir_sub, k)
        img_name_list.append(img_name)
        feature_list.append(get_feature(img_name, metric_type))
    return_feature = torch.cat(feature_list)
    avg_feature = torch.mean(return_feature, dim=0)
    difference = return_feature - avg_feature
    difference_norm = torch.norm(difference, dim=1)
    index = torch.argmin(difference_norm)
    return_img = img_name_list[index]
    return return_img



def compute_metrics(src_dir, src_path, training_num=10):

    metric_type = 'sscd'

    given_type_id = src_dir
    return_dir_name = ''
    current_threshold = 100.0
    for threshold_name in os.listdir(given_type_id):
        threshold= float(threshold_name.split('_')[0])
        if threshold < current_threshold:
            current_threshold = threshold
            return_dir_name = threshold_name
    threshold_dir = os.path.join(given_type_id, return_dir_name)
    if len(os.listdir(threshold_dir)) < training_num:
        raise ValueError(f'{threshold_dir} with less than {training_num} classes')
    
    for sub_id in os.listdir(threshold_dir):
        sub_dir = os.path.join(threshold_dir, sub_id)
        mean_img_id = find_average_id(sub_dir, metric_type)
        
        tar_path = mean_img_id.replace('Cluster', 'Cluster_Filtered')
        if not os.path.exists(os.path.dirname(tar_path)):
            os.makedirs(os.path.dirname(tar_path))
        shutil.copyfile(mean_img_id, tar_path)

    src_dir = src_dir.replace('Cluster', 'Cluster_Filtered')

    given_type_id = src_dir
    save_path = given_type_id.replace('Cluster_Filtered', 'Cluster_Extracted')
    measure_extracted_untargeted(given_type_id, src_path, save_path, metric_type=metric_type, compute_type = 'AS-')


def best_cluster_measurement(src_dir, src_path):
    metric_type = 'sscd'

    given_type_id = src_dir

    subcount=500
    save_path = given_type_id.replace('Generator_Output', f'Generator_Output_Best_Clustering/{subcount}')
    measure_extracted_untargeted(given_type_id, src_path, save_path, metric_type=metric_type, compute_type = 'AS-', max_count=subcount)
    subcount=10
    save_path = given_type_id.replace('Generator_Output', f'Generator_Output_Best_Clustering/{subcount}')
    measure_extracted_untargeted(given_type_id, src_path, save_path, metric_type=metric_type, compute_type = 'AS-', max_count=subcount)


def main(src_dataset_mode = 0,
    training_num = 10,
    per_step = 50,
    mode = 0,
    model_mode = "v1.4", 
    checkpointing_steps=1000,
    gen_num=100,
    cfg=5.0,
    fix_term=0.0,
    baseline=False,
    best_case=False):
   
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
            print(f'Skipping {current_style}')
            continue
        instance_dir = f"{instance_src}/{current_style}"
        if baseline:
            src_path = f"{Prefix}/Generator_Output/{src_dataset_path}/baseline_{training_mode_list[mode]}_{training_num}_{per_step}_{model_mode}_cfg_{cfg}/{current_style}"
            tar_path = f"{Prefix}/Cluster_Output/{src_dataset_path}/{max_wait_time}/baseline_{training_mode_list[mode]}_{training_num}_{per_step}_{model_mode}_cfg_{cfg}/{current_style}"
        else:
            src_path = f"{Prefix}/Generator_Output/{src_dataset_path}/{training_mode_list[mode]}_{training_num}_{per_step}_{model_mode}_cfg_{cfg}_fix_{fix_term}/{current_style}"
            tar_path = f"{Prefix}/Cluster_Output/{src_dataset_path}/{max_wait_time}/{training_mode_list[mode]}_{training_num}_{per_step}_{model_mode}_cfg_{cfg}_fix_{fix_term}/{current_style}"
        if best_case:
            best_cluster_measurement(src_path, instance_dir)
        else:
            if not os.path.exists(tar_path):
                os.makedirs(tar_path)
            cluster_progress(src_path, gen_num*5, tar_path, training_num)
            compute_metrics(tar_path, instance_dir, training_num)



if __name__ == '__main__':
    main(training_num = 10, mode=0, per_step=200, checkpointing_steps=2000, cfg=3.0, fix_term=-0.02, baseline=False)
    main(training_num = 10, mode=0, per_step=200, checkpointing_steps=2000, cfg=3.0, baseline=True)
    main(training_num = 10, mode=1, per_step=200, checkpointing_steps=2000, cfg=5.0, fix_term=0.0, baseline=False)
    main(training_num = 10, mode=1, per_step=200, checkpointing_steps=2000, cfg=3.0, baseline=True)

