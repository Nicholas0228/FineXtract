# FineXtract: Extracting Training Data from Personalized Diffusion Models

FineXtract extracts the fine-tuning dataset based on the guided denoising process from pretrained DM to fine-tuned DM. Below we provide the process from fine-tuning to extracting on VanGogh's paintings.

### Requirements

See file "requirements.txt".

### Training Personalized DM

Run the following code to fine-tune a DM based on VanGogh's paintings.

```
Python Trainer.py
```

The checkpoints will be saved in "checkpoints""

### Extracting Training images

Run the following code to generate images using the fine-tuned checkpoints.

```
Python Generator.py
```

where the output images are shown in "Generator_Output" dir.

### Running the Clustering Alg.

Run the following code to cluster generate images.

```
Python Cluster.py
```

The final result is shown in "Cluster_Extracted dir"
