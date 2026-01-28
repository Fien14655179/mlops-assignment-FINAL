import h5py
import numpy as np
import pytest
import torch

from ml_core.data.loader import get_dataloaders
from ml_core.data.pcam import PCAMDataset


class TestPCAMPipeline:
    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """Creates dummy H5 files with names expected by loader.py"""
        # dit is een pytest fixture: pytest roept dit automatisch aan en geeft de output door
        # tmp_path is een tijdelijke map die pytest voor je aanmaakt (dus geen echte data nodig)
        # idee: we maken nep PCAM-bestanden zodat we onze pipeline kunnen testen zonder de echte dataset
        for split in ["train", "valid"]:
            # we bouwen de bestandsnamen precies zoals loader.py ze verwacht
            # zodat de loader later "denkt" dat dit echte PCAM files zijn
            x_path = tmp_path / f"camelyonpatch_level_2_split_{split}_x.h5"
            y_path = tmp_path / f"camelyonpatch_level_2_split_{split}_y.h5"

            # we openen twee H5 files: eentje voor images (x) en eentje voor labels (y)
            # "w" betekent: nieuw bestand maken / overschrijven als het al bestaat
            with h5py.File(x_path, "w") as f_x, h5py.File(y_path, "w") as f_y:
                # we maken datasets binnen die H5 files
                # x krijgt 100 samples van 96x96 met 3 channels (RGB)
                x_ds = f_x.create_dataset("x", (100, 96, 96, 3), dtype="float32")
                # y krijgt 100 labels in de vorm zoals PCAM ze opslaat (100,1,1,1)
                y_ds = f_y.create_dataset("y", (100, 1, 1, 1), dtype="int64")

                # we maken eerst “normale” data: alles gevuld met 128 (grijs)
                # dit is handig als standaard baseline
                data = np.full((100, 96, 96, 3), 128.0, dtype="float32")

                # nu voegen we expres rare / extreme gevallen toe om de code te testen
                # sample 0: één pixel heeft een extreem hoge waarde (1e5)
                # dit test of we correct clippen voordat we naar uint8 casten
                data[0, 0, 0, :] = 1e5

                # sample 1: volledig zwart beeld (alles 0)
                # dit is een outlier die je filter misschien weg moet gooien
                data[1, :, :, :] = 0.0

                # sample 2: volledig wit beeld (alles 255)
                # ook dit is een outlier die mean-based filtering zou moeten herkennen
                data[2, :, :, :] = 255.0

                # nu maken we labels
                # eerst alles 0 (negatieve class)
                labels = np.zeros((100, 1, 1, 1), dtype="int64")

                # laatste 20 samples zetten we op 1 (positieve class)
                # dit creëert een 80/20 split zodat we weighted sampling kunnen testen
                labels[80:] = 1

                # schrijf de gemaakte arrays naar de H5 datasets
                x_ds[:] = data
                y_ds[:] = labels

        # pytest geeft tmp_path terug zodat de tests weten waar de nepdata staat
        return tmp_path

    def test_numerical_stability(self, mock_data_dir):
        """Checks if 1e5 values are clipped to 255 before becoming uint8."""
        # we pakken de paths naar de train-split files
        x_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_x.h5")
        y_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_y.h5")

        # maak de dataset aan zonder filtering zodat we sample 0 zeker kunnen lezen
        ds = PCAMDataset(x_p, y_p, filter_data=False)

        # pak het eerste sample: daar zit expres een 1e5 pixel in
        img, _ = ds[0]

        # check: als je correct clipt naar 255 vóór uint8 cast
        # dan mag geen enkele waarde groter zijn dan 255
        assert (
            img.max() <= 255
        ), "Image values > 255 found. Did you forget to clip before uint8 cast?"

    def test_heuristic_filtering(self, mock_data_dir):
        """Checks if mean-based filtering drops the black/white outlier samples."""
        # weer dezelfde train files
        x_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_x.h5")
        y_p = str(mock_data_dir / "camelyonpatch_level_2_split_train_y.h5")

        # nu zetten we filter_data=True zodat de dataset outliers moet verwijderen
        ds = PCAMDataset(x_p, y_p, filter_data=True)

        # we hebben 100 samples gemaakt
        # maar sample 1 (zwart) en sample 2 (wit) moeten eruit
        # dus we verwachten 98 over te houden
        assert (
            len(ds.indices) == 98
        ), f"Filtering failed. Expected 98 samples, got {len(ds.indices)}"

    def test_dataloader_output_logic(self, mock_data_dir):
        """Verifies shapes, types, and label squeezing."""
        # we maken een minimale config zoals get_dataloaders verwacht
        # batch_size=4 zodat we makkelijk de shapes kunnen checken
        # num_workers=0 zodat dit stabiel draait in tests (geen multiprocessing issues)
        config = {
            "data": {"data_path": str(mock_data_dir), "batch_size": 4, "num_workers": 0}
        }

        # haal de train loader op (valid loader boeit hier minder)
        train_loader, _ = get_dataloaders(config)

        # haal één batch uit de loader
        images, labels = next(iter(train_loader))

        # check: images moeten PyTorch format zijn: (batch, channels, height, width)
        assert images.shape == (4, 3, 96, 96)

        # check: labels moeten long zijn omdat classification labels meestal int64 zijn in torch
        assert labels.dtype == torch.long

        # check: labels moeten 1D zijn (batch_size,)
        # want vaak komt PCAM uit een vorm (batch,1,1,1) en je moet dat “squeezen”
        assert labels.dim() == 1, "Labels should be squeezed to 1D (Batch size,)"

    def test_weighted_sampling(self, mock_data_dir):
        """Verifies WeightedRandomSampler balances the 80/20 split."""
        # hier testen we of weighted sampling de class imbalance compenseert
        # batch_size=40 zodat we genoeg samples hebben om te zien of het echt helpt
        config = {
            "data": {
                "data_path": str(mock_data_dir),
                "batch_size": 40,
                "num_workers": 0,
            }
        }

        # haal de loader op
        train_loader, _ = get_dataloaders(config)

        # pak één batch en kijk naar de labels
        _, labels = next(iter(train_loader))

        # tel hoeveel positives (label 1) er in de batch zitten
        positives = (labels == 1).sum().item()

        # zonder sampler zou je bij 80/20 ongeveer 8 positives verwachten in 40
        # met sampler verwacht je meer balans, dus duidelijk meer dan 12 is een makkelijke check
        assert (
            positives > 12
        ), f"WeightedSampler might not be working. Only {positives}/40 are class 1."
