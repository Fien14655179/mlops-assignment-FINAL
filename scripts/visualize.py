import argparse
import yaml
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from ml_core.models.mlp import LateFusionMLP
from ml_core.data.dataset import TCGAMultimodalDataset

def visualize(model_path, config_path, output_path):
    # we lezen eerst de config in zodat we precies weten hoe de data en het model zijn ingesteld
    # dit zorgt dat deze visualisatie altijd klopt met de training-instellingen
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # we maken een dataset voor de test-split
    # hiermee pakken we dezelfde soort input als tijdens evaluatie
    dataset = TCGAMultimodalDataset("test", config)

    # DataLoader maakt batches zodat we niet alles tegelijk in het geheugen hoeven te laden
    # shuffle=False want we hoeven de volgorde niet random te maken voor deze plot
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # we bouwen het model op basis van de config
    # dit moet dezelfde architectuur zijn als waarmee je getraind hebt
    model = LateFusionMLP(config["model"])

    # we laden de opgeslagen gewichten in het model
    # map_location='cpu' betekent: we kunnen dit ook draaien zonder GPU
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # eval-modus: geen dropout en geen training-gedrag
    model.eval()

    # hier gaan we de verborgen representaties (features) verzamelen
    features_list = []
    # hier verzamelen we de labels zodat we later kunnen kleuren per class
    labels_list = []

    print("Extracting features...")
    # tijdens feature-extractie willen we geen gradients berekenen
    # dat scheelt veel geheugen en tijd
    with torch.no_grad():
        # elk item uit de loader geeft visuele input, tekst input, label, en nog een extra veld (bijv. patient id)
        for x_vis, x_txt, y, _ in loader:
            # hier kiezen we welke input we gebruiken
            # als we multimodal zijn, plakken we de twee embeddings naast elkaar
            if config["model"]["visual_dim"] > 0 and config["model"]["text_dim"] > 0:
                x = torch.cat((x_vis, x_txt), dim=1)
            # als alleen visual bestaat, gebruiken we alleen x_vis
            elif config["model"]["visual_dim"] > 0:
                x = x_vis
            # anders gebruiken we alleen tekst
            else:
                x = x_txt
            
            # we willen nu niet de eind-output (class scores) visualiseren
            # maar de 'verborgen laag' waar het model representaties leert
            # daarom voeren we handmatig de eerste lagen uit
            hidden = model.layers[0](x)
            hidden = model.layers[1](hidden)
            
            # we slaan de verborgen representaties op als numpy arrays
            # later plakken we alles samen in één grote matrix
            features_list.append(hidden.numpy())

            # labels ook opslaan zodat we straks weten welke punten bij elkaar horen
            labels_list.append(y.numpy())

    # van lijstjes naar één grote matrix / vector
    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)

    print(f"Running UMAP on shape {features.shape}...")
    # UMAP maakt van hoge-dimensie features (bijv. 128) een 2D embedding
    # zodat je clusters en structuur visueel kunt zien
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features)

    # nu maken we een scatterplot van de 2D embedding
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1], 
        # hue=labels betekent: kleur per label/class
        hue=labels, 
        # tab20 geeft veel verschillende kleuren, handig voor meerdere classes
        palette="tab20", 
        legend="full",
        # puntgrootte klein houden omdat er veel punten zijn
        s=15
    )

    # titel zodat duidelijk is wat je ziet
    plt.title("Latent Space Visualization (UMAP)")

    # legenda naar de zijkant zetten zodat het plotvlak niet te vol wordt
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)

    # tight_layout zodat alles netjes in het figuur past
    plt.tight_layout()
    
    # opslaan naar bestand
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # dit stuk zorgt dat je dit script vanuit de terminal kunt runnen met argumenten
    parser = argparse.ArgumentParser()

    # pad naar het .pt of .pth bestand met model weights
    parser.add_argument("--model_path", required=True)

    # pad naar de yaml config
    parser.add_argument("--config", required=True)

    # outputbestand voor de plot (standaard naam als je niks meegeeft)
    parser.add_argument("--output", default="umap_plot.png")

    # parse de argumenten van de command line
    args = parser.parse_args()

    # run de visualisatie met de ingestelde paden
    visualize(args.model_path, args.config, args.output)