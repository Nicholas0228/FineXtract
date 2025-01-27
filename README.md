# Leveraging Model Guidance to Extract Training Data from Personalized Diffusion Models

FineXtract is a framework designed to extract the fine-tuning dataset from personalized diffusion models (DMs), leveraging the guided denoising process between pretrained and fine-tuned DMs. In this document, we detail the steps for fine-tuning and dataset extraction using Monet's paintings as an example.

## Preparations

### Downloading SSCD Models

Run following command to download pretrained SSCD models for evaluating extractions. This command will create the directory `pretrainedmodels` and download the pretrained SSCD model required for extraction.

```
mkdir -p pretrainedmodels
wget -P pretrainedmodels https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt
```

### Requirements

Install the necessary Python dependencies listed in `requirements.txt`.

```
pip install -r requirements.txt
```

## Training Personalized Diffusion Model

Run the following code to fine-tune a DM based on Monet's paintings.

```
Python Trainer.py
```

The fine-tuned model checkpoints will be saved in the `checkpoints` directory.

## Extracting Training Images

Run the following code to generate images using the fine-tuned checkpoints.

```
Python Generator.py
```

The generated images will be saved in the `Generator_Output` directory.

## Running the Clustering Algorithm

Run the following code to cluster the generated images based on their similairty, producing a set of images that best represent the fine-tuned dataset.

```
Python Cluster.py
```

The final clustering results will be saved in the `Cluster_Extracted` directory.
