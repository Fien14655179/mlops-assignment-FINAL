import pytest
import torch
from ml_core.models import MLP


class TestMLPImplementation:
    # deze class bevat tests die controleren of ons MLP-model zich gedraagt
    # zoals we verwachten, zonder naar de interne details te hoeven kijken

    @pytest.fixture
    def sample_config(self):
        # dit is een vaste set instellingen die we in meerdere tests hergebruiken
        # het stelt voor: een afbeelding met 3 kleurkanalen van 96x96 pixels
        # twee verborgen lagen met 64 en 32 neuronen
        # en uiteindelijk 2 mogelijke klassen als output
        return {"input_shape": [3, 96, 96], "hidden_units": [64, 32], "num_classes": 2}

    def test_forward_pass(self, sample_config):
        """Verifies the model flattens input and outputs correct logit shapes."""
        # we maken een nieuw model aan met de voorbeeldconfiguratie
        model = MLP(**sample_config)

        # we maken nep-inputdata
        # 8 staat voor het aantal samples in de batch
        # daarna komt de vorm van één input (3, 96, 96)
        x = torch.randn(8, *sample_config["input_shape"])

        # we voeren de data door het model
        output = model(x)

        # hier checken we of de outputvorm klopt
        # per input (8 stuks) verwachten we 2 getallen
        # die staan voor de logits van de twee klassen
        # als dit faalt, is de input waarschijnlijk niet goed “platgemaakt”
        assert output.shape == (
            8,
            2,
        ), f"Expected (8, 2), got {output.shape}. Did you flatten?"

    def test_backprop(self, sample_config):
        """Ensures weights update, verifying the computational graph isn't broken."""
        # opnieuw maken we een model aan
        model = MLP(**sample_config)

        # we kiezen een simpele optimizer die de gewichten stap voor stap aanpast
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # we pakken één parameter uit het model
        # dit is gewoon een voorbeeldgewicht dat we gaan volgen
        param = next(model.parameters())

        # we slaan de beginwaarde op
        # zodat we later kunnen checken of deze echt veranderd is
        initial_val = param.clone()

        # opnieuw maken we nep-inputdata
        # nu met batchgrootte 2
        x = torch.randn(2, *sample_config["input_shape"])

        # dit zijn de bijbehorende labels
        # 0 en 1 betekenen: sample 1 hoort bij klasse 0, sample 2 bij klasse 1
        y = torch.tensor([0, 1], dtype=torch.long)

        # we berekenen de loss door de output van het model
        # te vergelijken met de juiste labels
        loss = torch.nn.functional.cross_entropy(model(x), y)

        # hier wordt de fout “teruggestuurd” door het netwerk
        # zodat elk gewicht weet hoe het moet veranderen
        loss.backward()

        # de optimizer gebruikt deze informatie om de gewichten echt aan te passen
        optimizer.step()

        # nu checken we of het gewicht veranderd is
        # als dit niet zo is, dan werkt de terugkoppeling niet goed
        assert not torch.equal(
            initial_val, param
        ), "Weights did not update. Is the graph broken?"
