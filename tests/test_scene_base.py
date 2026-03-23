from witwin.core.scene import SceneBase


def test_scene_base_uses_add_mutators_without_with_aliases():
    scene = SceneBase(device="cpu")
    structure = object()
    source = object()
    monitor = object()

    returned = scene.add_structure(structure).add_source(source).add_monitor(monitor)

    assert returned is scene
    assert scene.structures == [structure]
    assert scene.sources == [source]
    assert scene.monitors == [monitor]
    assert not hasattr(scene, "with_structure")
    assert not hasattr(scene, "with_source")
    assert not hasattr(scene, "with_monitor")
