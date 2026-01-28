def test_imports():
    """
    Verifies that the core components are accessible through the package
    structure as defined in the __init__.py files.
    """

    # deze test kijkt niet naar logica of berekeningen
    # hij checkt puur of de structuur van het project klopt
    # met andere woorden: kun je alles importeren zoals bedoeld?

    # hier proberen we twee dingen te importeren uit ml_core.data
    # als de __init__.py daar goed staat, lukt dit zonder fouten
    from ml_core.data import PCAMDataset, get_dataloaders

    # deze asserts zijn simpel maar belangrijk
    # ze bevestigen dat de imports echt iets opleveren
    # en niet per ongeluk None zijn door een misconfiguratie
    assert PCAMDataset is not None
    assert get_dataloaders is not None

    # nu testen we hetzelfde voor het model-gedeelte
    # MLP moet beschikbaar zijn via ml_core.models
    from ml_core.models import MLP

    # als dit faalt, weet je dat je package-structuur niet klopt
    assert MLP is not None

    # hier checken we of de trainer correct geëxporteerd is
    # dit is vaak een centrale klasse in een ML-pipeline
    from ml_core.solver import Trainer

    # opnieuw: bestaan is genoeg voor deze test
    assert Trainer is not None

    # tot slot testen we de utility-functies
    # dit zijn vaak helpers die overal in het project gebruikt worden
    from ml_core.utils import (
        ExperimentTracker,
        load_config,
        seed_everything,
        setup_logger,
    )

    # elke utility moet bereikbaar zijn
    # zo weet je zeker dat externe code hier ook op kan vertrouwen
    assert ExperimentTracker is not None
    assert load_config is not None
    assert seed_everything is not None
    assert setup_logger is not None